# This function reads a QAP problem from a .mat file and prepares it for solving.
function read_QAP_mat(FILE_NAME::String)
    QAPdata = matread(FILE_NAME)
    A = Float32.(QAPdata["A"])
    B = Float32.(QAPdata["B"])
    S = Float32.(QAPdata["S"])
    T = Float32.(QAPdata["T"])
    n = size(A, 1)
    
    println("QAP problem information: nRow = ", 2*n, ", nCol = ", n^2)
    A_gpu = CuMatrix(A)
    B_gpu = CuMatrix(B)
    S_gpu = CuMatrix(S)
    T_gpu = CuMatrix(T)

    Q = QAP_Q_operator_gpu(A_gpu, B_gpu, S_gpu, T_gpu, n)
    c = CUDA.zeros(Float32, n^2)
    ee = ones(Float32, n)
    Id = spdiagm(ones(Float32, n))
    A = sparse(vcat(kron(ee', Id), kron(Id, ee')))

    A_gpu = CuSparseMatrixCSR(A)
    AT_gpu = CuSparseMatrixCSR(A')
    b = CuVector(ones(Float32, 2 * n))
    l = CUDA.zeros(Float32, n^2)
    u = Inf32 * CUDA.ones(Float32, n^2)
    # Convert to standard QP_info format
    standard_qp = QP_info_gpu(Q, c, A_gpu, AT_gpu, b, copy(b), l, u, 0.0f0, [])

    return standard_qp
end

# This function formulates a LASSO problem from a sparse matrix A, vector b, and regularization parameter lambda.
function formulate_LASSO_from_A_b_lambda(A::SparseMatrixCSC, b::Vector{Float32}, lambda::Float32)
    ## LASSO: min 0.5 ||Ax-b||^2 + λ ||x||_1
    n_org = size(A, 2)
    ATb = A' * b
    lambda_vec = lambda * ones(Float32, n_org)
    println("LASSO problem information: nRow = ", 0, ", nCol = ", n_org)
    c = -ATb
    c0 = 0.5f0 * norm(b)^2
    Q = LASSO_Q_operator_gpu(CuSparseMatrixCSR(A), CuSparseMatrixCSR(A'))
    A_csr = CuSparseMatrixCSR(sparse([], [], Float32[], 0, n_org))
    AT_csr = CuSparseMatrixCSR(sparse([], [], Float32[], n_org, 0))
    c_gpu = CuVector{Float32}(c)
    lp_AL = CuVector{Float32}(zeros(Float32, 0))
    lp_AU = CuVector{Float32}(zeros(Float32, 0))
    l = CuVector{Float32}(-Inf32 * ones(Float32, n_org))
    u = CuVector{Float32}(Inf32 * ones(Float32, n_org))
    lambda_gpu = CuVector{Float32}(lambda_vec)

    standard_qp = QP_info_gpu(Q, c_gpu, A_csr, AT_csr, lp_AL, lp_AU, l, u, c0, lambda_gpu)

    # Return the modified qp
    return standard_qp
end

# This function reads a LASSO problem from a .mat file and prepares it for solving.
function formulate_LASSO_from_mat(file::String)
    ## LASSO: min 0.5 ||Ax-b||^2 + λ ||x||_1, we take the value of λ = 1e-3 * ||A' b||_∞ for LIBSVM instances
    LASSO = matread(file)
    A = sparse(Float32.(LASSO["A"]))
    b = vec(Float32.(LASSO["b"]))
    n_org = size(A, 2)
    ATb = A' * b
    lambda_val = 1e-3f0 * norm(ATb, Inf)
    lambda = lambda_val * ones(Float32, n_org)
    c = -ATb
    c0 = 0.5f0 * norm(b)^2
    Q = LASSO_Q_operator_gpu(CuSparseMatrixCSR(A), CuSparseMatrixCSR(A'))
    A_csr = CuSparseMatrixCSR(sparse([], [], Float32[], 0, n_org))
    AT_csr = CuSparseMatrixCSR(sparse([], [], Float32[], n_org, 0))
    c_gpu = CuVector{Float32}(c)
    lp_AL = CuVector{Float32}(zeros(Float32, 0))
    lp_AU = CuVector{Float32}(zeros(Float32, 0))
    l = CuVector{Float32}(-Inf32 * ones(Float32, n_org))
    u = CuVector{Float32}(Inf32 * ones(Float32, n_org))
    lambda_gpu = CuVector{Float32}(lambda)

    standard_qp = QP_info_gpu(Q, c_gpu, A_csr, AT_csr, lp_AL, lp_AU, l, u, c0, lambda_gpu)

    # Return the modified qp
    return standard_qp
end

# This function performs scaling on the QP problem data and returns the scaling information.
function scaling!(qp::QP_info_gpu, params::HPRQP_parameters)
    m, n = size(qp.A)
    row_norm = ones(Float32, m)
    col_norm = ones(Float32, n)

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL .== -Inf32] .= 0.0f0
    AU_nInf[qp.AU .== Inf32] .= 0.0f0
    
    initial_norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf)
    initial_norm_c = norm(qp.c, Inf)

    scaling_info = Scaling_info_gpu(CuVector(copy(qp.l)), CuVector(copy(qp.u)), CuVector(row_norm), CuVector(col_norm), 1.0f0, 1.0f0, 1.0f0, 1.0f0, initial_norm_b, initial_norm_c)

    if params.use_bc_scaling
        b_scale = max(1.0f0, norm(max.(abs.(AL_nInf), abs.(AU_nInf))))
        c_scale = max(1.0f0, norm(qp.c))

        qp.AL ./= b_scale
        qp.AU ./= b_scale
        qp.c ./= c_scale
        qp.lambda ./= c_scale
        qp.l ./= b_scale # Assuming l and u are scaled with b
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0f0
        scaling_info.c_scale = 1.0f0
    end

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL .== -Inf32] .= 0.0f0
    AU_nInf[qp.AU .== Inf32] .= 0.0f0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(qp.c)
    scaling_info.row_norm = CuVector(row_norm)
    scaling_info.col_norm = CuVector(col_norm)

    return scaling_info
end

function mean(x::Vector{Float32})
    return sum(x) / length(x)
end

# This function performs power iteration to estimate the largest eigenvalue of a matrix AAT.
function power_iteration_A_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR, max_iterations::Int=5000, tolerance::Float32=1e-4f0)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), Float32, m)) .+ 1e-8f0 # Initial random vector
    q = CUDA.zeros(Float32, m)
    ATq = CUDA.zeros(Float32, n)
    lambda_max = 1.0f0
    error = 1.0f0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1.0f0, AT, q, 0.0f0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1.0f0, A, ATq, 0.0f0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q      # error 
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
function power_iteration_Q_QAP_gpu(Q::QAP_Q_operator_gpu, max_iterations::Int=5000, tolerance::Float32=1e-4f0)
    seed = 1
    n = Q.n^2
    z = CuVector(randn(Random.MersenneTwister(seed), Float32, n)) .+ 1e-8f0 # Initial random vector
    q = CUDA.zeros(Float32, n)
    lambda_max = 1.0f0
    temp1 = CUDA.zeros(Float32, n)
    error = 1.0f0
    for i in 1:max_iterations
        q .= z
        if CUDA.norm(q) < 1e-15f0
            println("Power iteration failed to converge.")
            return 1.0f0
        end
        q ./= CUDA.norm(q)
        QAP_Qmap!(q, z, temp1, Q)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q      # error 
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
function power_iteration_Q_LASSO_gpu(Q::LASSO_Q_operator_gpu, max_iterations::Int=5000, tolerance::Float32=1e-4f0)
    seed = 1
    m, n = size(Q.A)
    z = CuVector(randn(Random.MersenneTwister(seed), Float32, n)) .+ 1e-7f0 # Initial random vector
    q = CUDA.zeros(Float32, n)
    lambda_max = 1.0f0
    temp1 = CUDA.zeros(Float32, m)
    error = 1.0f0
    for i in 1:max_iterations
        q .= z
        if CUDA.norm(q) < 1e-7f0
            println("Power iteration failed to converge.")
            return 1.0f0
        end
        q ./= CUDA.norm(q)
        LASSO_Qmap!(q, z, temp1, Q) # Assuming a LASSO-specific map function exists
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q      # error 
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
        t_start_scale = time()
        println("SCALING ... ")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
        println(@sprintf("SCALING time: %.2f seconds", time() - t_start_scale))
    else
        println("SCALING: OFF")
        # Still call scaling to initialize scaling_info, but with scaling off
        params_no_scale = deepcopy(params)
        params_no_scale.use_bc_scaling = false
        scaling_info_gpu = scaling!(standard_qp_gpu, params_no_scale)
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
    csv_file = result_path * "HPRQP_result_f32.csv"

    # redirect the output to a file
    log_path = result_path * "HPRQP_log_f32.txt"

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
            geomean_time = exp(mean(log.(Float32.(timelist) .+ 10.0f0))) - 10.0f0
            geomean_time_4 = exp(mean(log.(Float32.(time3list) .+ 10.0f0))) - 10.0f0
            geomean_time_6 = exp(mean(log.(Float32.(time6list) .+ 10.0f0))) - 10.0f0
            geomean_iter = exp(mean(log.(Float32.(iterlist) .+ 10.0f0))) - 10.0f0
            geomean_iter_4 = exp(mean(log.(Float32.(iter4list) .+ 10.0f0))) - 10.0f0
            geomean_iter_6 = exp(mean(log.(Float32.(iter6list) .+ 10.0f0))) - 10.0f0
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
function solve_LASSO_from_A_b(A::SparseMatrixCSC, b::Vector{Float32}, params::HPRQP_parameters)
    CUDA.device!(params.device_number)
    setup_start = time()
    println("FORMULATING ... ")
    # Assuming params.lambda is Float32 or will be converted correctly inside
    lambda_f32 = Float32(params.lambda)
    standard_qp_gpu = formulate_LASSO_from_A_b_lambda(A, b, lambda_f32)
    formulate_time = time() - setup_start
    println(@sprintf("FORMULATING time: %.2f seconds", formulate_time))

    scale_time = 0.0
    if params.use_bc_scaling
        t_start_scale = time()
        println("SCALING ... ")
        scaling_info_gpu = scaling!(standard_qp_gpu, params)
        scale_time = time() - t_start_scale
        println(@sprintf("SCALING time: %.2f seconds", scale_time))
    else
        println("SCALING: OFF")
        params_no_scale = deepcopy(params)
        params_no_scale.use_bc_scaling = false
        scaling_info_gpu = scaling!(standard_qp_gpu, params_no_scale)
    end

    setup_time = formulate_time + scale_time
    results = solve(standard_qp_gpu, scaling_info_gpu, params)
    println(@sprintf("Total time: %.2fs", setup_time + results.time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results

end