using LinearAlgebra, SparseArrays, Random, MAT
import HPRQP_QAP_LASSO
using Printf

"""
Generates a random LASSO problem instance with Float32 precision.

Args:
    m (Int): Number of rows in matrix A.
    n (Int): Number of columns in matrix A.
    sparsity (Float32): The sparsity level of matrix A. Defaults to 1.0f-4.
    seed (Int): The random seed for reproducibility. Defaults to 1.

Returns:
    Tuple{SparseMatrixCSC{Float32, Int}, Vector{Float32}}: A tuple containing the sparse matrix A and the vector b.
"""
function gen_random_LASSO(m::Int, n::Int, sparsity::Float32=1.0f-4, seed::Int=1)
    Random.seed!(seed)
    
    # Generate a sparse matrix A with Float32 elements.
    A = sprandn(Float32, m, n, sparsity)
    
    # Generate a random solution vector x with Float32 elements.
    x = randn(Float32, n)
    
    # Create sparsity in the solution vector by setting half of its entries to zero.
    idx = randperm(n)
    zero_idx = idx[1:Int(round(n / 2))]
    x[zero_idx] .= 0.0f0 # Use Float32 zero
    
    # Generate the right-hand side vector b with a small Float32 epsilon.
    b = A * x .+ 1.0f-6 # Use Float32 epsilon
    
    return A, b
end

# --- Configuration ---
save_to_mat = false # Set to true to save the instance to a .mat file
solve = true        # Set to true to solve the instance

# --- Problem Generation ---
m = 50000
n = 100000
sparsity = 1.0f-4 # Use Float32 for sparsity literal

println(@sprintf("Start to generate a random LASSO instance with m = %d, n = %d, sparsity = %.2e", m, n, sparsity))
t_start = time()

# Generate the LASSO problem with Float32 precision.
A, b = gen_random_LASSO(m, n, sparsity)

# Calculate lambda using Float32 precision.
# The norm calculation will produce a Float32 result since A and b are Float32.
lambda::Float32 = 1.0f-3 * norm(A' * b, Inf)

println(@sprintf("Generate LASSO time: %.2f seconds", time() - t_start))

# --- Save Instance (Optional) ---
if save_to_mat
    t_start = time()
    println("Start to save the LASSO instance (A, b) to .mat file")
    # The MAT library will handle the conversion of Float32 arrays.
    mat_data = Dict("A" => A, "b" => b)
    matwrite("random_LASSO_f32.mat", mat_data)
    println(@sprintf("Save LASSO instance to .mat file time: %.2f seconds", time() - t_start))
end

# --- Solve Instance (Optional) ---
if solve
    # Configure solver parameters.
    params = HPRQP_QAP_LASSO.HPRQP_parameters()
    params.max_iter = typemax(Int32)
    params.time_limit = 3600
    params.stoptol = 1.0f-8 # Use Float32 for the stopping tolerance
    params.device_number = 0
    params.problem_type = "LASSO"
    params.lambda = lambda # Assign the Float32 lambda value

    # Call the solver.
    # The function is expected to handle Float32 inputs for A and b.
    HPRQP_QAP_LASSO.solve_LASSO_from_A_b(A, b, params)
end
