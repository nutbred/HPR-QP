using CUDA, LinearAlgebra, SparseArrays, Random, Printf, MAT
using CUDA.CUSPARSE
using HPRQP_QAP_LASSO

function gen_random_QAP(n::Int, seed::Int=1)
    Random.seed!(seed)
    points_x = rand(n)
    points_y = rand(n)

    A = sqrt.((points_x .- points_x') .^ 2 .+ (points_y .- points_y') .^ 2)
    A .= (A + A') / 2  # Make A symmetric

    B = rand(n, n)
    B .= B - diagm(diag(B))  # Ensure B is not diagonal
    B .= (B + B') / 2  # Make B symmetric

    return A, B
end

function solve_QAP(A, B, file_path, solve=true)
    n = size(A, 1)
    Ascale = max(1.0, norm(A, 2))
    Bscale = max(1.0, norm(B, 2))
    A ./= Ascale
    B ./= Bscale
    ee = ones(n)

    println("copy A and B to GPU")
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    println("successfully copied A and B to GPU")

    t_start = time()
    println("start to compute eigenvalues and eigenvectors of A and B")
    DA, VA = eigen(A_gpu)
    idxA = sortperm(DA, rev=true)
    DA = DA[idxA]
    VA = VA[:, idxA]
    DB, VB = eigen(B_gpu)
    idxB = sortperm(DB, rev=false)
    DB = DB[idxB]
    VB = VB[:, idxB]
    println("successfully computed eigenvalues and eigenvectors of A and B")
    t_end = time()
    println("time taken: $(t_end - t_start) seconds")

    t_start = time()
    println("start to compute cost matrix and linear constraints, inv")
    DA = Vector(DA)
    DB = Vector(DB)
    costMat = DA * DB'
    Id = spdiagm(0 => ones(n))
    AE = sparse(vcat(kron(ee', Id), kron(Id, ee')))
    AE2 = AE[1:end-1, :]
    idx = []
    for i = 1:n
        push!(idx, n * (i - 1) + i)
        push!(idx, n * (i - 1) + i + 1)
    end
    idx = idx[1:end-1]

    Basis = AE2[:, idx]
    cB = costMat[idx]

    y = (Basis') \ cB
    y[abs.(y).<1e-15] .= 0.0
    push!(y, 0.0)
    println("successfully computed cost matrix and linear constraints, inv")
    t_end = time()
    println("time taken: $(t_end - t_start) seconds")

    ss = y[1:n]
    tt = y[n+1:end]

    println("start to compute S and T")
    t_start = time()
    S = VA * (CuVector(ss) .* VA')
    S .= (S + S') / 2  # Make S symmetric
    T = VB * (CuVector(tt) .* VB')
    T .= (T + T') / 2  # Make T symmetric
    println("successfully computed S and T")
    t_end = time()
    println("time taken: $(t_end - t_start) seconds")

    # save A B S T to .mat file
    S = Array(S)
    T = Array(T)

    t_start = time()
    println("start to save A, B, S, T to .mat file")
    mat_data = Dict("A" => A, "B" => B, "S" => S, "T" => T)
    matwrite(file_path, mat_data)
    println("successfully saved A, B, S, T to .mat file")
    t_end = time()
    println("time taken: $(t_end - t_start) seconds")

    if solve
        params = HPRQP_QAP_LASSO.HPRQP_parameters()
        params.max_iter = typemax(Int32)
        params.time_limit = 3600
        params.stoptol = 1e-8
        params.device_number = 0
        params.problem_type = "QAP"

        result = HPRQP_QAP_LASSO.run_file(file_path, params)
    end
end

A, B = gen_random_QAP(128)  # Generate a random QAP instance with n=128
# You can input your own A and B matrices in other ways, e.g., from a file or other sources.

file_path = "test_128.mat"  # Replace with the path where you want to save the .mat files
solve = true  # Set to true if you want to solve the instance using HPRQP_QAP_LASSO
solve_QAP(A, B, file_path, solve)