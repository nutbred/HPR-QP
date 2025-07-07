# This function reads a QAP problem from a .mat file and prepares it for solving.
function read_QAP_mat(FILE_NAME::String)
    QAPdata = matread(FILE_NAME)
    A = QAPdata["A"]
    B = QAPdata["B"]
    S = QAPdata["S"]
    T = QAPdata["T"]
    n = size(A, 1)
    
    println("QAP problem information: nRow = ", 2*n, ", nCol = ", n^2)
    A_gpu = CuMatrix(A)
    B_gpu = CuMatrix(B)
    S_gpu = CuMatrix(S)
    T_gpu = CuMatrix(T)

    Q = QAP_Q_operator_gpu(A_gpu, B_gpu, S_gpu, T_gpu, n)
    c = CUDA.zeros(Float64, n^2)
    ee = ones(Float64, n)
    Id = spdiagm(ones(Float64, n))
    A = sparse(vcat(kron(ee', Id), kron(Id, ee')))

    A_gpu = CuSparseMatrixCSR(A)
    AT_gpu = CuSparseMatrixCSR(A')
    b = CuVector(ones(Float64, 2 * n))
    l = CUDA.zeros(Float64, n^2)
    u = Inf * CUDA.ones(Float64, n^2)
    # Convert to standard QP_info format
    standard_qp = QP_info_gpu(Q, c, A_gpu, AT_gpu, b, copy(b), l, u, 0.0, [])

    return standard_qp
end

# This function formulates a LASSO problem from a sparse matrix A, vector b, and regularization parameter lambda.
function formulate_LASSO_from_A_b_lambda(A::SparseMatrixCSC, b::Vector{Float64}, lambda::Float64)
    ## LASSO: min 0.5 ||Ax-b||^2 + λ ||x||_1
    n_org = size(A, 2)
    ATb = A' * b
    lambda = lambda * ones(n_org)
    println("LASSO problem information: nRow = ", 0, ", nCol = ", n_org)
    c = -ATb
    c0 = 0.5 * norm(b)^2
    Q = LASSO_Q_operator_gpu(CuSparseMatrixCSR(A), CuSparseMatrixCSR(A'))
    A = CuSparseMatrixCSR(sparse([], [], Float64[], 0, n_org))
    AT = CuSparseMatrixCSR(sparse([], [], Float64[], n_org, 0))
    c = CuVector{Float64}(c)
    lp_AL = CuVector{Float64}(zeros(0))
    lp_AU = CuVector{Float64}(zeros(0))
    l = CuVector{Float64}(-Inf * ones(n_org))
    u = CuVector{Float64}(Inf * ones(n_org))
    lambda = CuVector{Float64}(lambda)

    standard_qp = QP_info_gpu(Q, c, A, AT, lp_AL, lp_AU, l, u, c0, lambda)

    # Return the modified qp
    return standard_qp
end

# This function reads a LASSO problem from a .mat file and prepares it for solving.
function formulate_LASSO_from_mat(file::String)
    ## LASSO: min 0.5 ||Ax-b||^2 + λ ||x||_1, we take the value of λ = 1e-3 * ||A' b||_∞ for LIBSVM instances
    LASSO = matread(file)
    A = sparse(LASSO["A"])
    b = reshape(LASSO["b"], size(A, 1))
    n_org = size(A, 2)
    ATb = A' * b
    lambda = 1e-3 * norm(ATb, Inf) * ones(n_org)
    c = -ATb
    c0 = 0.5 * norm(b)^2
    Q = LASSO_Q_operator_gpu(CuSparseMatrixCSR(A), CuSparseMatrixCSR(A'))
    A = CuSparseMatrixCSR(sparse([], [], Float64[], 0, n_org))
    AT = CuSparseMatrixCSR(sparse([], [], Float64[], n_org, 0))
    c = CuVector{Float64}(c)
    lp_AL = CuVector{Float64}(zeros(0))
    lp_AU = CuVector{Float64}(zeros(0))
    l = CuVector{Float64}(-Inf * ones(n_org))
    u = CuVector{Float64}(Inf * ones(n_org))
    lambda = CuVector{Float64}(lambda)

    standard_qp = QP_info_gpu(Q, c, A, AT, lp_AL, lp_AU, l, u, c0, lambda)

    # Return the modified qp
    return standard_qp
end

# This function performs scaling on the QP problem data and returns the scaling information.
function scaling!(qp::QP_info_gpu, params::HPRQP_parameters)
    m, n = size(qp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info = Scaling_info_gpu(CuVector(copy(qp.l)), CuVector(copy(qp.l)), CuVector(row_norm), CuVector(col_norm), 1, 1, 1, 1, norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), norm(qp.c, Inf))
    if params.use_bc_scaling
        b_scale = max(1, norm(min.(qp.AL, qp.AU)), norm(qp.c))
        c_scale = b_scale

        qp.AL ./= b_scale
        qp.AU ./= b_scale
        qp.c ./= c_scale
        qp.lambda ./= c_scale
        qp.l ./= b_scale
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(qp.c)
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    return scaling_info
end

function mean(x::Vector{Float64})
    return sum(x) / length(x)
end

# This function performs power iteration to estimate the largest eigenvalue of a matrix AAT.
function power_iteration_A_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR, max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = CUDA.norm(q) / (CUDA.norm(z) + lambda_max)
        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# This function performs power iteration to estimate the largest eigenvalue of Q in a QAP problem.
function power_iteration_Q_QAP_gpu(Q::QAP_Q_operator_gpu, max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    n = Q.n^2
    z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    temp1 = CUDA.zeros(Float64, n)
    error = 1.0
    for i in 1:max_iterations
        q .= z
        if CUDA.norm(q) < 1e-15
            println("Power iteration failed to converge.")
            return 1.0
        end
        q ./= CUDA.norm(q)
        QAP_Qmap!(q, z, temp1, Q)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = CUDA.norm(q) / (CUDA.norm(z) + lambda_max)
        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# This function performs power iteration to estimate the largest eigenvalue of Q in a LASSO problem.
function power_iteration_Q_LASSO_gpu(Q::LASSO_Q_operator_gpu, max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(Q.A)
    z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    temp1 = CUDA.zeros(Float64, m)
    error = 1.0
    for i in 1:max_iterations
        q .= z
        if CUDA.norm(q) < 1e-15
            println("Power iteration failed to converge.")
            return 1.0
        end
        q ./= CUDA.norm(q)
        QAP_Qmap!(q, z, temp1, Q)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = CUDA.norm(q) / (CUDA.norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# This function runs the HPR-QP algorithm on a given file with specified parameters.
function run_file(FILE_NAME::String, params::HPRQP_parameters)
    CUDA.device!(params.device_number)
    t_start = time()
    println("READING FILE ... ", FILE_NAME)
    if endswith(FILE_NAME, ".mat") && params.problem_type == "QAP"
        standard_qp_gpu = read_QAP_mat(FILE_NAME)
    elseif endswith(FILE_NAME, ".mat") && params.problem_type == "LASSO"
        standard_qp_gpu = formulate_LASSO_from_mat(FILE_NAME)
    else
        println("we only support QAP and LASSO instances in .mat format")
        error("the file format is: ", FILE_NAME[end-2:end], ", but the problem type is: ", params.problem_type)
    end
    read_time = time() - t_start
    println(@sprintf("READING FILE time: %.2f seconds", read_time))

    setup_start = time()
    if params.use_bc_scaling
        t_start = time()
        println("SCALING ... ")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
        println(@sprintf("SCALING time: %.2f seconds", time() - t_start))
    else
        println("SCALING: OFF")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
    end
    setup_time = time() - setup_start

    results = solve(standard_qp_gpu, scaling_info_gpu, params)
    println(@sprintf("Total time: %.2fs", read_time + setup_time + results.time),
        @sprintf("  read time = %.2fs", read_time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results
end

# This function runs the HPR-QP algorithm on a dataset of files and saves the results to a CSV file.
function run_dataset(data_path::String, result_path::String, params::HPRQP_parameters)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRQP_result.csv"

    # redirect the output to a file
    log_path = result_path * "HPRQP_log.txt"

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time3list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        powerlist = Vector{Any}(result_table.power_time[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time3list = []
        iter6list = []
        time6list = []
        powerlist = []
    end

    warm_up_done = false
    for i = 1:length(files)
        file = files[i]
        if occursin(".mat", file) && !(file in namelist)
            FILE_NAME = data_path * file
            if params.warm_up && !warm_up_done
                max_iter = params.max_iter
                params.max_iter = 200
                warm_up_done = true
                println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                params.max_iter = max_iter
                all_time = time() - t_start_all
                println("warm up time: ", all_time)
                println("warm up ends ----------------------------------------------------------------------------------------------------------")
            end

            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter4list, results.iter_4)
                push!(time3list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
                push!(powerlist, results.power_time)
            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time3list,
                iter_6=iter6list,
                time_6=time6list,
                power_time=powerlist,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time3list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, ""])

            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_3 = count(x -> x < params.time_limit, time3list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_3, "", solved_6, ""])

            CSV.write(csv_file, result_table)
        end
    end

    close(io)
end

# it's used in demo_LASSO.jl
function solve_LASSO_from_A_b(A::SparseMatrixCSC, b::Vector{Float64}, params::HPRQP_parameters)
    CUDA.device!(params.device_number)
    t_start = time()
    setup_start = time()
    println("FORMULATING ... ")
    standard_qp_gpu = formulate_LASSO_from_A_b_lambda(A, b, params.lambda)
    println(@sprintf("FORMULATING time: %.2f seconds", time() - t_start))

    if params.use_bc_scaling
        t_start = time()
        println("SCALING ... ")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
        println(@sprintf("SCALING time: %.2f seconds", time() - t_start))
    else
        println("SCALING: OFF")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
    end

    setup_time = time() - setup_start
    results = solve(standard_qp_gpu, scaling_info_gpu, params)
    println(@sprintf("Total time: %.2fs", setup_time + results.time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results

end