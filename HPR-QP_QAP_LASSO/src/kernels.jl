## kernels for LASSO

# This function computes the Qx = A'*(A*x) for LASSO problem on GPU.
@inline function QAP_Qmap!(x::CuVector{Float32}, Qx::CuVector{Float32}, temp1::CuVector{Float32}, Q::LASSO_Q_operator_gpu)
    # Perform sparse matrix-vector multiplication for A*x
    CUDA.CUSPARSE.mv!('N', 1.0f0, Q.A, x, 0.0f0, temp1, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # Perform sparse matrix-vector multiplication for A'*(A*x)
    CUDA.CUSPARSE.mv!('T', 1.0f0, Q.A, temp1, 0.0f0, Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

CUDA.@fastmath @inline function update_zxw_kernel!(lambda::CuDeviceVector{Float32},
                                                 dw::CuDeviceVector{Float32},
                                                 dx::CuDeviceVector{Float32},
                                                 w_bar::CuDeviceVector{Float32},
                                                 w::CuDeviceVector{Float32},
                                                 z_bar::CuDeviceVector{Float32},
                                                 x_bar::CuDeviceVector{Float32},
                                                 x_hat::CuDeviceVector{Float32},
                                                 last_x::CuDeviceVector{Float32},
                                                 x::CuDeviceVector{Float32},
                                                 Qw::CuDeviceVector{Float32},
                                                 ATy::CuDeviceVector{Float32},
                                                 c::CuDeviceVector{Float32},
                                                 tempv::CuDeviceVector{Float32},
                                                 sigma::Float32,
                                                 fact1::Float32,
                                                 fact2::Float32,
                                                 Halpern_fact1::Float32,
                                                 Halpern_fact2::Float32,
                                                 n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        lambda_sigma = lambda[i] * sigma
        # Soft-thresholding operation
        x_bar[i] = (z_bar[i] < -lambda_sigma) ? (z_bar[i] + lambda_sigma) : ((z_bar[i] > lambda_sigma) ? (z_bar[i] - lambda_sigma) : 0.0f0)
        x_hat[i] = 2.0f0 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]

        # Halpern iteration for x
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        # Update w using extrapolation
        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
        w[i] = Halpern_fact1 * w[i] + Halpern_fact2 * (2.0f0 * w_bar[i] - w[i])
        dw[i] = w_bar[i] - w[i]
    end
    return
end

## kernels for QAP
# This function computes the Qx = 2*(AXB-SX-XT) for QAP problem on GPU.
@inline function QAP_Qmap!(x::CuVector{Float32}, Qx::CuVector{Float32}, temp1::CuVector{Float32}, Q::QAP_Q_operator_gpu)
    n = Q.n
    X = reshape(x, n, n)
    QX = reshape(Qx, n, n)
    TMP = reshape(temp1, n, n)
    # Compute 2 * A * X * B
    mul!(TMP, Q.A, X)
    mul!(QX, TMP, Q.B, 2.0f0, 0.0f0)
    # Compute QX = QX - 2 * S * X
    mul!(QX, Q.S, X, -2.0f0, 1.0f0)
    # Compute QX = QX - 2 * X * T
    mul!(QX, X, Q.T, -2.0f0, 1.0f0)
end


CUDA.@fastmath @inline function update_zxw1_kernel!(dx::CuDeviceVector{Float32},
                                                  w_bar::CuDeviceVector{Float32},
                                                  w::CuDeviceVector{Float32},
                                                  z_bar::CuDeviceVector{Float32},
                                                  x_bar::CuDeviceVector{Float32},
                                                  x_hat::CuDeviceVector{Float32},
                                                  last_x::CuDeviceVector{Float32},
                                                  x::CuDeviceVector{Float32},
                                                  Qw::CuDeviceVector{Float32},
                                                  ATy::CuDeviceVector{Float32},
                                                  c::CuDeviceVector{Float32},
                                                  tempv::CuDeviceVector{Float32},
                                                  l::CuDeviceVector{Float32},
                                                  u::CuDeviceVector{Float32},
                                                  sigma::Float32,
                                                  fact1::Float32,
                                                  fact2::Float32,
                                                  Halpern_fact1::Float32,
                                                  Halpern_fact2::Float32,
                                                  n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        # Projection onto the bounds [l, u]
        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2.0f0 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        # Halpern iteration for x
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
    end
    return
end

function update_zxw1_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float32, Halpern_fact2::Float32)
    QAP_Qmap!(ws.w, ws.Qw, ws.temp1, qp.Q)
    fact2 = 1.0f0 / (1.0f0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0f0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_kernel!(dy::CuDeviceVector{Float32},
                          y_bar::CuDeviceVector{Float32},
                          y::CuDeviceVector{Float32},
                          last_y::CuDeviceVector{Float32},
                          s::CuDeviceVector{Float32},
                          AL::CuDeviceVector{Float32},
                          AU::CuDeviceVector{Float32},
                          Ax::CuDeviceVector{Float32},
                          fact1::Float32,
                          fact2::Float32,
                          Halpern_fact1::Float32,
                          Halpern_fact2::Float32,
                          m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        s[i] = Ax[i] - fact1 * y[i]
        # Projection for dual variable y
        y_bar[i] = s[i] < AL[i] ? (AL[i] - s[i]) : (s[i] > AU[i] ? (AU[i] - s[i]) : 0.0f0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        # Halpern iteration for y
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2.0f0 * y_bar[i] - y[i])
    end
    return
end

function update_y_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float32, Halpern_fact2::Float32)
    QAP_Qmap!(ws.w_bar, ws.Qw_bar, ws.temp1, qp.Q)
    axpby_gpu!(1.0f0, ws.tempv, -ws.sigma, ws.Qw_bar, ws.tempv, ws.n)
    CUDA.CUSPARSE.mv!('N', 1.0f0, qp.A, ws.tempv, 0.0f0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0f0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
end

function update_w2_kenel!(dw::CuDeviceVector{Float32},
                          ATdy::CuDeviceVector{Float32},
                          w::CuDeviceVector{Float32},
                          w_bar::CuDeviceVector{Float32},
                          last_w::CuDeviceVector{Float32},
                          last_ATy::CuDeviceVector{Float32},
                          ATy::CuDeviceVector{Float32},
                          ATy_bar::CuDeviceVector{Float32},
                          fact::Float32,
                          Halpern_fact1::Float32,
                          Halpern_fact2::Float32,
                          n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = w_bar[i] + fact * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        # Halpern iteration for w and ATy
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2.0f0 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2.0f0 * ATy_bar[i] - ATy[i])
    end
    return
end

function update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float32, Halpern_fact2::Float32)
    CUDA.CUSPARSE.mv!('T', 1.0f0, ws.A, ws.y_bar, 0.0f0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact = ws.sigma / (1.0f0 + ws.sigma * ws.lambda_max_Q)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w2_kenel!(ws.dw, ws.ATdy, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, fact, Halpern_fact1, Halpern_fact2, ws.n)
end

## kernels for compute residuals

function compute_Rd_kernel!(ATy::CuDeviceVector{Float32},
                            z::CuDeviceVector{Float32},
                            c::CuDeviceVector{Float32},
                            Qx::CuDeviceVector{Float32},
                            Rd::CuDeviceVector{Float32},
                            col_norm::CuDeviceVector{Float32},
                            c_scale::Float32,
                            n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        # Compute dual residual
        Rd[i] = Qx[i] + c[i] - ATy[i] - z[i]
        # Apply scaling
        scale_fact = col_norm[i] * c_scale
        Rd[i] *= scale_fact
        Qx[i] *= scale_fact
        ATy[i] *= scale_fact
    end
    return
end

function compute_Rd_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    if ws.m > 0
        CUDA.CUSPARSE.mv!('T', 1.0f0, ws.A, ws.y_bar, 0.0f0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
end

function compute_Rp_kernel!(Rp::CuDeviceVector{Float32},
                            AL::CuDeviceVector{Float32},
                            AU::CuDeviceVector{Float32},
                            Ax::CuDeviceVector{Float32},
                            row_norm::CuDeviceVector{Float32},
                            b_scale::Float32,
                            m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        # Compute primal residual
        Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : (Ax[i] > AU[i] ? (AU[i] - Ax[i]) : 0.0f0)
        # Apply scaling
        scale_fact = row_norm[i] * b_scale
        Rp[i] *= scale_fact
        Ax[i] *= scale_fact
    end
    return
end

function compute_Rp_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1.0f0, ws.A, ws.x_bar, 0.0f0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
end

function compute_err_lu_kernel!(dx::CuDeviceVector{Float32},
                                x::CuDeviceVector{Float32},
                                l::CuDeviceVector{Float32},
                                u::CuDeviceVector{Float32},
                                col_norm::CuDeviceVector{Float32},
                                b_scale::Float32,
                                n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        # Compute bound violation error
        dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0f0)
        dx[i] *= b_scale / col_norm[i]
    end
    return
end

function axpby_kernel!(a::Float32, x::CuDeviceVector{Float32},
                       b::Float32, y::CuDeviceVector{Float32},
                       z::CuDeviceVector{Float32}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float32, x::CuArray{Float32},
                    b::Float32, y::CuArray{Float32},
                    z::CuArray{Float32}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

## function for update sigma

@inline function f_dev(x, a, b, c, d)
    return a * x + b / x + c * x^2 / (1.0f0 + d * x)
end

function golden(
    a_p::Float32, b_p::Float32, c_p::Float32, d_p::Float32;
    lo::Float32=eps(Float32),
    hi::Float32=1.0f12,
    tol::Float32=1.0f-6, # Adjusted tolerance for Float32
    maxiter::Int=200
)
    # golden ratio constant
    φ = (sqrt(5.0f0) - 1.0f0) / 2.0f0
    a = lo
    b = hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    f_c = f_dev(c, a_p, b_p, c_p, d_p)
    f_d = f_dev(d, a_p, b_p, c_p, d_p)

    for i in 1:maxiter
        if f_d < f_c
            a, c, f_c = c, d, f_d
            d = a + φ * (b - a)
            f_d = f_dev(d, a_p, b_p, c_p, d_p)
        else
            b, d, f_d = d, c, f_c
            c = b - φ * (b - a)
            f_c = f_dev(c, a_p, b_p, c_p, d_p)
        end
        if (b - a) < tol
            break
        end
    end

    x_sol = 0.5f0 * (a + b)
    return x_sol
end