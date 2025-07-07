using LinearAlgebra, SparseArrays, Random, MAT
import HPRQP_QAP_LASSO
using Printf

function gen_random_LASSO(m::Int, n::Int, sparsity::Float64=1e-4, seed::Int=1)
    Random.seed!(seed)
    A = sprandn(m, n, sparsity)  # Sparse matrix with given sparsity
    x = randn(n)  # Random solution vector
    idx = randperm(n)  # Random permutation of indices
    zero_idx = idx[1:Int(round(n / 2))]
    x[zero_idx] .= 0.0  # Set half of the entries to zero to create sparsity
    b = A * x .+ 1e-6  # Generate the right-hand side vector
    return A, b
end

save_to_mat = false # Set to true if you want to save the instance to a .mat file
solve = true # Set to true if you want to solve the instance using HPRQP_QAP_LASSO
# generate random LASSO instance and solve it using HPRQP
m = 50000
n = 100000
sparsity = 1e-4
println(@sprintf("Start to generate a random LASSO instance with m = %d, n = %d, sparsity = %.2e", m, n, sparsity))
t_start = time()
A, b = gen_random_LASSO(m, n, sparsity)
lambda = 1e-3 * norm(A' * b, Inf)
println(@sprintf("generate LASSO time: %.2f seconds", time() - t_start))

if save_to_mat
    t_start = time()
    println("Start to save the LASSO instance (A, b) to .mat file")
    mat_data = Dict("A" => A, "b" => b)
    matwrite("random_LASSO.mat", mat_data)
    println(@sprintf("Save LASSO instance to .mat file time: %.2f seconds", time() - t_start))
end

if solve
    params = HPRQP_QAP_LASSO.HPRQP_parameters()
    params.max_iter = typemax(Int32)
    params.time_limit = 3600
    params.stoptol = 1e-8
    params.device_number = 0
    params.problem_type = "LASSO"
    params.lambda = lambda

    HPRQP_QAP_LASSO.solve_LASSO_from_A_b(A, b, params)
end