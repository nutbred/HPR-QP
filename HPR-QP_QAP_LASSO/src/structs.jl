# This struct stores the problem data for QAP.
mutable struct QAP_Q_operator_gpu
    A::CuMatrix{Float32}
    B::CuMatrix{Float32}
    S::CuMatrix{Float32}
    T::CuMatrix{Float32}
    n::Int
end

# This struct stores the problem data for LASSO.
mutable struct LASSO_Q_operator_gpu
    A::CuSparseMatrixCSR{Float32,Int32}
    AT::CuSparseMatrixCSR{Float32,Int32}
end

# This struct stores the problem data for QP.
mutable struct QP_info_gpu
    Q::Union{QAP_Q_operator_gpu,LASSO_Q_operator_gpu}
    c::CuVector{Float32}
    A::CuSparseMatrixCSR{Float32,Int32}
    AT::CuSparseMatrixCSR{Float32,Int32}
    AL::CuVector{Float32}
    AU::CuVector{Float32}
    l::CuVector{Float32}
    u::CuVector{Float32}
    obj_constant::Float32
    lambda::CuVector{Float32}
end

# This struct stores the scaling information.
mutable struct Scaling_info_gpu
    l_org::CuVector{Float32}
    u_org::CuVector{Float32}
    row_norm::CuVector{Float32}
    col_norm::CuVector{Float32}
    b_scale::Float32
    c_scale::Float32
    norm_b::Float32
    norm_c::Float32
    norm_b_org::Float32
    norm_c_org::Float32
end

# This struct contains parameters for the HPR-QP solver.
mutable struct HPRQP_parameters
    """
        stoptol::Float32
            Stopping tolerance for the optimization algorithm.
        sigma::Float32
            Penalty parameter or regularization parameter used in the algorithm.
        max_iter::Int
            Maximum number of iterations allowed.
        sigma_fixed::Bool
            Indicates whether the sigma parameter is fixed during optimization.
        time_limit::Float32
            Maximum allowed runtime (in seconds) for the algorithm.
        eig_factor::Float32
            Factor used to scale the maximum eigenvalue estimation.
        check_iter::Int
            Frequency (in iterations) to check for convergence or perform certain checks.
        warm_up::Bool
            Whether to perform a warm-up phase before the main optimization.
        problem_type::String
            Type of problem being solved (e.g., "LASSO", "QAP").
        print_frequency::Int
            Frequency (in iterations) for printing progress information.
        device_number::Int32
            Identifier for the computational device (e.g., GPU number 0 1 2 3).
        use_bc_scaling::Bool
            Whether to use box-constraint scaling in the algorithm.
        lambda::Float32
            Regularization parameter for LASSO problems.
    """
    stoptol::Float32
    sigma::Float32
    max_iter::Int
    sigma_fixed::Bool
    time_limit::Float32
    eig_factor::Float32
    check_iter::Int
    warm_up::Bool
    problem_type::String
    print_frequency::Int
    device_number::Int32
    use_bc_scaling::Bool
    lambda::Float32
    HPRQP_parameters() = new(1f-6, -1.0f0, typemax(Int32), false, 3600.0f0, 1.05f0, 100, false, "QAP", -1, 0, false, 0.1f0)
end

# This struct stores the residuals and other metrics during the HPR-QP algorithm.
mutable struct HPRQP_residuals
    err_Rp_org_bar::Float32
    err_Rd_org_bar::Float32
    is_updated::Bool
    KKTx_and_gap_org_bar::Float32
    primal_obj_bar::Float32
    rel_gap_bar::Float32
    dual_obj_bar::Float32
    # Define a default constructor
    HPRQP_residuals() = new()
end

# This struct stores the results of the HPR-QP algorithm.
mutable struct HPRQP_results
    iter::Int              # Total number of iterations performed.
    iter_4::Int            # Number of iterations to get 1e-4 (if applicable).
    iter_6::Int            # Number of iterations to get 1e-6 (if applicable).
    time::Float32          # Total computation time (seconds).
    time_4::Float32        # Computation time spent to get 1e-4 (seconds).
    time_6::Float32        # Computation time spent to get 1e-6 (seconds).
    power_time::Float32    # Time spent on eigenvalue estimation (seconds).
    primal_obj::Float32    # Final value of the primal objective function.
    residuals::Float32     # Final value of the residuals.
    gap::Float32           # Final duality gap.
    output_type::String    # Status or type of output (e.g., "OPTIMAL", "MAX_ITER", "TIME_LIMIT").
    x::Vector{Float32}     # Solution vector for the primal variables.
    y::Vector{Float32}     # Solution vector for the dual variables (equality/inequality constraints).
    z::Vector{Float32}     # Solution vector for the dual variables (bounds).
    w::Vector{Float32}     # Auxiliary variable vector.
    HPRQP_results() = new()
end

# This struct stores the workspace for the HPR-QP algorithm on GPU.
mutable struct HPRQP_workspace_gpu
    w::CuVector{Float32}
    w_hat::CuVector{Float32}
    w_bar::CuVector{Float32}
    dw::CuVector{Float32}
    x::CuVector{Float32}
    x_hat::CuVector{Float32}
    x_bar::CuVector{Float32}
    dx::CuVector{Float32}
    y::CuVector{Float32}
    y_hat::CuVector{Float32}
    y_bar::CuVector{Float32}
    dy::CuVector{Float32}
    z_bar::CuVector{Float32}
    Q::CuSparseMatrixCSR{Float32,Int32}
    A::CuSparseMatrixCSR{Float32,Int32}
    AT::CuSparseMatrixCSR{Float32,Int32}
    AL::CuVector{Float32}
    AU::CuVector{Float32}
    c::CuVector{Float32}
    l::CuVector{Float32}
    u::CuVector{Float32}
    Rp::CuVector{Float32}
    Rd::CuVector{Float32}
    m::Int
    n::Int
    sigma::Float32
    lambda_max_A::Float32
    lambda_max_Q::Float32
    Ax::CuVector{Float32}
    ATy::CuVector{Float32}
    ATy_bar::CuVector{Float32}
    ATdy::CuVector{Float32}
    QATdy::CuVector{Float32}
    s::CuVector{Float32}
    Qw::CuVector{Float32}
    Qw_hat::CuVector{Float32}
    Qw_bar::CuVector{Float32}
    Qx::CuVector{Float32}
    dQw::CuVector{Float32}
    last_x::CuVector{Float32}
    last_y::CuVector{Float32}
    last_Qw::CuVector{Float32}
    last_w::CuVector{Float32}
    last_ATy::CuVector{Float32}
    tempv::CuVector{Float32}
    diag_Q::CuVector{Float32}
    fact1::CuVector{Float32}
    fact2::CuVector{Float32}
    fact::CuVector{Float32}
    fact_M::CuVector{Float32}
    temp1::CuVector{Float32}
    HPRQP_workspace_gpu() = new()
end

# This struct stores the restart information for the HPR-QP algorithm.
mutable struct HPRQP_restart
    restart_flag::Int
    first_restart::Bool
    last_gap::Float32
    current_gap::Float32
    save_gap::Float32
    inner::Int
    step::Int
    sufficient::Int
    necessary::Int
    long::Int
    ratio::Int
    times::Int
    weighted_norm::Float32

    HPRQP_restart() = new()
end