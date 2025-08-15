### This file contains the main algorithm for solving convex composite quadratic programming problems using the HPR method on GPU
# when the explicit matrix form of Q is not avaliable.

# The package is used to solve convex composite quadratic programming with HPR method in the paper 
# HPR-QP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming
# The package is developed by Kaihuang Chen · Defeng Sun · Yancheng Yuan · Guojun Zhang · Xinyuan Zhao.

# Convex Composite Quadratic Programming problem formulation:
# 
#     minimize      (1/2) x' Q x + ϕ(x)
#     subject to    AL ≤ Ax ≤ AU
#
# where:
#   - Q is a symmetric positive semidefinite matrix (n x n)
#   - c is a vector (n)
#   - A is a constraint matrix (m x n)
#   - ϕ(x) is a proper, closed, and convex function, such as L1 regularization in LASSO
#   - x is the variable vector (n)

# This function computes the residuals for the HPR-QP algorithm on GPU.
function compute_residuals_gpu(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int,
)

    ### obj
    QAP_Qmap!(ws.x_bar, ws.Qx, ws.temp1, qp.Q)
    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (CUDA.dot(ws.c, ws.x_bar) + 0.5f0 * CUDA.dot(ws.x_bar, ws.Qx)) + qp.obj_constant

    res.dual_obj_bar = sc.b_scale * sc.c_scale *
                       (-0.5f0 * CUDA.dot(ws.x_bar, ws.Qx) + CUDA.dot(ws.z_bar, ws.x_bar)) + qp.obj_constant
    if ws.m > 0
        res.dual_obj_bar += sc.b_scale * sc.c_scale * CUDA.dot(ws.y_bar, ws.s)
    end

    if params.problem_type == "LASSO"
        ## compute the L1 norm term in LASSO 
        ws.tempv .= qp.lambda .* ws.x_bar
        l1_norm = CUDA.norm(ws.tempv, 1)
        res.primal_obj_bar += sc.b_scale * sc.c_scale * l1_norm
        res.dual_obj_bar += sc.b_scale * sc.c_scale * l1_norm
    end
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0f0 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))

    ### Rd
    compute_Rd_gpu!(ws, sc)

    res.err_Rd_org_bar = CUDA.norm(ws.Rd, Inf) / (1.0f0 + maximum([sc.norm_c_org, CUDA.norm(ws.ATdy, Inf), CUDA.norm(ws.Qx, Inf)]))

    ### Rp
    if ws.m > 0
        compute_Rp_gpu!(ws, sc)
        res.err_Rp_org_bar = CUDA.norm(ws.Rp, Inf) / (1.0f0 + max(sc.norm_b_org, CUDA.norm(ws.Ax, Inf)))
    else
        res.err_Rp_org_bar = 0.0f0
    end

    if params.problem_type == "QAP" && iter == 0
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_err_lu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, sc.col_norm, sc.b_scale, ws.n)
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, CUDA.norm(ws.dx, Inf))
    end
    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
end


# This function updates the value of sigma based on the current state of the algorithm.
function update_sigma(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        a = 0.0f0
        b = 0.0f0
        c = 0.0f0
        d = 0.0f0
        axpby_gpu!(1.0f0, ws.x_bar, -1.0f0, ws.last_x, ws.dx, ws.n)
        if ws.m > 0
            axpby_gpu!(1.0f0, ws.y_bar, -1.0f0, ws.last_y, ws.dy, ws.m)
            axpby_gpu!(1.0f0, ws.ATy_bar, -1.0f0, ws.last_ATy, ws.ATdy, ws.n)
            QAP_Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
            a += ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy) - 2.0f0 * CUDA.dot(ws.dQw, ws.ATdy)
        end
        axpby_gpu!(1.0f0, ws.w_bar, -1.0f0, ws.last_w, ws.dw, ws.n)
        QAP_Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)

        b += CUDA.dot(ws.dx, ws.dx)

        a += ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
        if ws.m > 0
            c += CUDA.dot(ws.ATdy, ws.QATdy)
            d += ws.lambda_max_Q
        end
        a = max(a, 1f-12)
        b = max(b, 1f-12)
        if ws.m > 0
            # min a * x + b / x + c * x^2 / (1 + d * x)
            sigma_new = golden(a, b, c, d; lo=1f-12, hi=1f12, tol=1f-13)
        else
            sigma_new = sqrt(b / a)
        end
        fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
        ws.sigma = exp(fact * log(sigma_new) + (1.0f0 - fact) * log(ws.sigma))
    end

end

# This function checks whether a restart is needed based on the current state of the algorithm.
function check_restart(restart_info::HPRQP_restart,
    iter::Int,
    check_iter::Int,
)

    restart_info.restart_flag = 0
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.weighted_norm = restart_info.current_gap
        end
    else
        if rem(iter, check_iter) == 0

            if restart_info.current_gap <= 0.2f0 * restart_info.last_gap
                restart_info.sufficient += 1
                restart_info.restart_flag = 1
            end

            if (restart_info.current_gap <= 0.8f0 * restart_info.last_gap) && (restart_info.current_gap > 1.00f0 * restart_info.save_gap)
                restart_info.necessary += 1
                restart_info.restart_flag = 2
            end
            if restart_info.current_gap / restart_info.weighted_norm > 1f-1
                fact = 0.5f0
            else
                fact = 0.2f0
            end

            if restart_info.inner >= fact * iter
                restart_info.long += 1
                restart_info.restart_flag = 3
            end
            restart_info.save_gap = restart_info.current_gap
        end
    end
end

# This function performs the restart.
function do_restart(restart_info, ws::HPRQP_workspace_gpu)
    if restart_info.restart_flag > 0
        ws.x .= ws.x_bar
        ws.w .= ws.w_bar
        ws.last_x .= ws.x_bar
        ws.last_w .= ws.w_bar
        if ws.m > 0
            ws.y .= ws.y_bar
            ws.last_y .= ws.y_bar
            CUDA.CUSPARSE.mv!('N', 1.0f0, ws.AT, ws.y_bar, 0.0f0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            ws.last_ATy .= ws.ATy_bar
            ws.ATy .= ws.ATy_bar
        end
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf32
        restart_info.times += 1
        restart_info.inner = 0
    end
end

# This function checks the stopping criteria for the HPR-QP algorithm.
function check_break(residuals::HPRQP_residuals,
    iter::Int,
    t_start_alg::Float32,
    params::HPRQP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

# This function collects the results after the HPR-QP algorithm has finished running.
function collect_results_gpu!(
    ws::HPRQP_workspace_gpu,
    residuals::HPRQP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float32,
    power_time::Float32,
)
    results = HPRQP_results()
    results.iter = iter
    results.time = Float32(time() - t_start_alg)
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    ### copy the results to the CPU ### 
    results.w = Vector(sc.b_scale * (ws.w_bar ./ sc.col_norm))
    results.x = Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y = Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z = Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))

    return results
end

# This function allocates the workspace for the HPR-QP algorithm on GPU.
function allocate_workspace_gpu(qp::QP_info_gpu,
    params::HPRQP_parameters,
    lambda_max_A::Float32,
    lambda_max_Q::Float32,
    scaling_info::Scaling_info_gpu,
)
    ws = HPRQP_workspace_gpu()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    if params.sigma == -1
        norm_b = scaling_info.norm_b
        norm_c = scaling_info.norm_c
        if norm_c > 1f-16 && norm_b > 1f-16 && norm_b < 1f16 && norm_c < 1f16
            ws.sigma = norm_b / norm_c
        else
            ws.sigma = 1.0f0
        end
    elseif params.sigma > 0
        ws.sigma = Float32(params.sigma)
    else
        error("Invalid sigma value: ", params.sigma, " should be positive or -1 for automatically chosen.")
    end
    println("initial sigma = ", ws.sigma)
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.w = CUDA.zeros(Float32, n)
    ws.w_hat = CUDA.zeros(Float32, n)
    ws.w_bar = CUDA.zeros(Float32, n)
    ws.dw = CUDA.zeros(Float32, n)
    ws.x = CUDA.zeros(Float32, n)
    ws.x_hat = CUDA.zeros(Float32, n)
    ws.x_bar = CUDA.zeros(Float32, n)
    ws.dx = CUDA.zeros(Float32, n)
    ws.y = CUDA.zeros(Float32, m)
    ws.y_hat = CUDA.zeros(Float32, m)
    ws.y_bar = CUDA.zeros(Float32, m)
    ws.dy = CUDA.zeros(Float32, m)
    ws.s = CUDA.zeros(Float32, m)
    ws.z_bar = CUDA.zeros(Float32, n)
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u
    if ws.m > 0
        ws.AL[ws.AL.==-Inf32] .= -1f20
        ws.AU[ws.AU.==Inf32] .= 1f20
    end
    ws.l[ws.l.==-Inf32] .= -1f20
    ws.u[ws.u.==Inf32] .= 1f20
    ws.Rp = CUDA.zeros(Float32, m)
    ws.Rd = CUDA.zeros(Float32, n)
    ws.ATy = CUDA.zeros(Float32, n)
    ws.ATy_bar = CUDA.zeros(Float32, n)
    ws.ATdy = CUDA.zeros(Float32, n)
    ws.QATdy = CUDA.zeros(Float32, n)
    ws.Ax = CUDA.zeros(Float32, m)
    ws.Qw = CUDA.zeros(Float32, n)
    ws.Qw_bar = CUDA.zeros(Float32, n)
    ws.Qw_hat = CUDA.zeros(Float32, n)
    ws.Qx = CUDA.zeros(Float32, n)
    ws.dQw = CUDA.zeros(Float32, n)
    ws.last_x = CUDA.zeros(Float32, n)
    ws.last_y = CUDA.zeros(Float32, m)
    ws.last_Qw = CUDA.zeros(Float32, n)
    ws.last_w = CUDA.zeros(Float32, n)
    ws.last_ATy = CUDA.zeros(Float32, n)
    ws.tempv = CUDA.zeros(Float32, n)
    ws.fact1 = CUDA.zeros(Float32, n)
    ws.fact2 = CUDA.zeros(Float32, n)
    ws.fact = CUDA.zeros(Float32, n)
    ws.fact_M = CUDA.zeros(Float32, n)
    if params.problem_type == "LASSO"
        ws.temp1 = CUDA.zeros(Float32, qp.Q.A.dims[1])
    elseif params.problem_type == "QAP"
        ws.temp1 = CUDA.zeros(Float32, n)
    end
    return ws
end

# This function initializes the restart information for the HPR-QP algorithm.
function initialize_restart(params::HPRQP_parameters)
    restart_info = HPRQP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf32
    restart_info.current_gap = Inf32
    restart_info.last_gap = Inf32
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.ratio = 0
    restart_info.restart_flag = 0
    restart_info.weighted_norm = Inf32
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end

# This function computes the M norm for the HPR-QP algorithm on GPU.
function compute_M_norm_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu)
    M_1 = 0.0f0
    M_2 = 1.0f0 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
    QAP_Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
    M_1 += ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
    M_2 -= 2.0f0 * CUDA.dot(ws.dQw, ws.dx)
    M_3 = 0.0f0

    if ws.m > 0
        M_1 += ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
        CUDA.CUSPARSE.mv!('N', 1.0f0, ws.AT, ws.dy, 0.0f0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        if typeof(qp.Q) <: Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
            QAP_Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
        elseif typeof(qp.Q) <: CuSparseMatrixCSR
            ws.QATdy .= qp.diag_Q .* ws.ATdy
        end
        M_1 -= 2.0f0 * ws.sigma * CUDA.dot(ws.dQw, ws.ATdy)
        M_2 += 2.0f0 * CUDA.dot(ws.ATdy, ws.dx)
        M_3 += (ws.sigma * ws.sigma) / (1.0f0 + ws.sigma * ws.lambda_max_Q) * CUDA.dot(ws.ATdy, ws.QATdy)  # sGS term
    end
    M_2 += max(M_1, 0.0f0)
    M_norm = max(M_2, 0.0f0) + max(M_3, 0.0f0)
    if min(M_1, M_2, M_3) < -1f-8
        println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3, negative M norm due to numerical instability, consider increasing eig_factor")
    end
    return sqrt(M_norm)
end

# This function is the main solver function for the HPR-QP algorithm on GPU.
function solve(qp::QP_info_gpu,
    scaling_info::Scaling_info_gpu,
    params::HPRQP_parameters
)
    m, n = size(qp.A)

    ### power iteration to estimate lambda_max ###
    CUDA.synchronize()
    t_start_alg = Float32(time())

    println("ESTIMATING MAXIMUM EIGENVALUES ...")
    if m > 0
        lambda_max_A = power_iteration_A_gpu(qp.A, qp.AT) * params.eig_factor
    else
        lambda_max_A = 0.0f0
    end
    if params.problem_type == "LASSO"
        lambda_max_Q = power_iteration_Q_LASSO_gpu(qp.Q) * params.eig_factor
    elseif params.problem_type == "QAP"
        lambda_max_Q = power_iteration_Q_QAP_gpu(qp.Q) * params.eig_factor
    else
        error("Invalid problem type: ", params.problem_type, " should be LASSO or QAP.")
    end
    CUDA.synchronize()
    power_time = Float32(time() - t_start_alg)
    println(@sprintf("ESTIMATING MAXIMUM EIGENVALUES time = %.2f seconds", power_time))
    println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max_A))
    println(@sprintf("estimated maximum eigenvalue of Q = %.2e", lambda_max_Q))

    ### Initialization ###
    residuals = HPRQP_residuals()

    restart_info = initialize_restart(params)

    ws = allocate_workspace_gpu(qp, params, lambda_max_A, lambda_max_Q, scaling_info)

    println("HPRQP SOLVER starts...")
    println(" iter      errRp        errRd         p_obj          d_obj           gap          sigma        time")

    iter_4 = 0
    time_4 = 0.0f0
    iter_6 = 0
    time_6 = 0.0f0
    first_4 = true
    first_6 = true

    check_iter = params.check_iter
    for iter = 0:params.max_iter
        if params.print_frequency == -1
            print_yes = ((rem(iter, print_step(iter)) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        elseif params.print_frequency > 0
            print_yes = ((rem(iter, params.print_frequency) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        else
            error("Invalid print_frequency: ", params.print_frequency, ". It should be a positive integer or -1 for automatic printing.")
        end

        if rem(iter, check_iter) == 0 || print_yes
            residuals.is_updated = true
            compute_residuals_gpu(ws, qp, scaling_info, residuals, params, iter)
        else
            residuals.is_updated = false
        end

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        check_restart(restart_info, iter, check_iter)

        ### update sigma ###
        update_sigma(params, restart_info, ws, qp)

        ### restart if needed ###
        do_restart(restart_info, ws)

        ### print the log ##
        if print_yes || (status != "CONTINUE")
            print(@sprintf("%5.0f", iter),
                @sprintf("    %3.2e", residuals.err_Rp_org_bar),
                @sprintf("      %3.2e", residuals.err_Rd_org_bar),
                @sprintf("      %7.6e", residuals.primal_obj_bar),
                @sprintf("      %7.6e", residuals.dual_obj_bar),
                @sprintf("      %3.2e", residuals.rel_gap_bar))
            # end
            print(@sprintf("      %3.2e", ws.sigma),
                @sprintf("      %6.2f", time() - t_start_alg))
            println()
        end


        ### collect results and return ###
        if residuals.KKTx_and_gap_org_bar < 1f-4 && first_4
            time_4 = Float32(time() - t_start_alg)
            iter_4 = iter
            first_4 = false
            println("KKT < 1e-4 at iter = ", iter)
        end
        if residuals.KKTx_and_gap_org_bar < 1f-6 && first_6
            time_6 = Float32(time() - t_start_alg)
            iter_6 = iter
            first_6 = false
            println("KKT < 1e-6 at iter = ", iter)
        end
        if status != "CONTINUE"
            if status == "OPTIMAL"
                println("The instance is solved, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            elseif status == "MAX_ITER"
                println("The maximum number of iterations is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            elseif status == "TIME_LIMIT"
                println("The time limit is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            end
            results = collect_results_gpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time)
            results.output_type = status
            results.time_4 = time_4 == 0.0f0 ? results.time : time_4
            results.iter_4 = iter_4 == 0 ? iter : iter_4
            results.time_6 = time_6 == 0.0f0 ? results.time : time_6
            results.iter_6 = iter_6 == 0 ? iter : iter_6
            return results
        end

        ## main iteatrion 
        Halpern_fact1 = 1.0f0 / (restart_info.inner + 2.0f0)
        Halpern_fact2 = 1.0f0 - Halpern_fact1

        if params.problem_type == "LASSO"
            QAP_Qmap!(ws.w, ws.Qw, ws.temp1, qp.Q)
            fact2 = 1.0f0 / (1.0f0 + ws.sigma * ws.lambda_max_Q)
            fact1 = 1.0f0 - fact2
            @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw_kernel!(qp.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
        elseif params.problem_type == "QAP"
            update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_y_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)
        end

        if restart_info.restart_flag > 0
            restart_info.last_gap = compute_M_norm_gpu!(ws, qp)
        end
        if rem(iter + 1, check_iter) == 0
            restart_info.current_gap = compute_M_norm_gpu!(ws, qp)
        end

        restart_info.inner += 1
    end

end