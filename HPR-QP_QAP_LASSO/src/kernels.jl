## kernels for LASSO

# This function computes the Qx = A'*(A*x) for LASSO problem on GPU.
@inline function QAP_Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, temp1::CuVector{Float64}, Q::LASSO_Q_operator_gpu)
    CUDA.CUSPARSE.mv!('N', 1, Q.A, x, 0, temp1, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, Q.AT, temp1, 0, Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

CUDA.@fastmath @inline function update_zxw_kernel!(lambda::CuDeviceVector{Float64},
                                                    dw::CuDeviceVector{Float64},
                                                    dx::CuDeviceVector{Float64},
                                                    w_bar::CuDeviceVector{Float64},
                                                    w::CuDeviceVector{Float64},
                                                    z_bar::CuDeviceVector{Float64},
                                                    x_bar::CuDeviceVector{Float64},
                                                    x_hat::CuDeviceVector{Float64},
                                                    last_x::CuDeviceVector{Float64},
                                                    x::CuDeviceVector{Float64},
                                                    Qw::CuDeviceVector{Float64},
                                                    ATy::CuDeviceVector{Float64},
                                                    c::CuDeviceVector{Float64},
                                                    tempv::CuDeviceVector{Float64},
                                                    sigma::Float64,
                                                    fact1::Float64,
                                                    fact2::Float64,
                                                    Halpern_fact1::Float64,
                                                    Halpern_fact2::Float64,
                                                    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        lambda_sigma = lambda[i] * sigma
        x_bar[i] = (z_bar[i] < -lambda_sigma) ? (z_bar[i] + lambda_sigma) : ((z_bar[i] > lambda_sigma) ? (z_bar[i] - lambda_sigma) : 0.0)
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
        w[i] = Halpern_fact1 * w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        dw[i] = w_bar[i] - w[i]
    end
    return
end

## kernels for QAP
# This function computes the Qx = 2*(AXB-SX-XT) for QAP problem on GPU.
@inline function QAP_Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, temp1::CuVector{Float64}, Q::QAP_Q_operator_gpu)
    n = Q.n
    X = reshape(x, n, n)
    QX = reshape(Qx, n, n)
    TMP = reshape(temp1, n, n)
    mul!(TMP, Q.A, X)
    mul!(QX, TMP, Q.B, 2.0, 0.0)
    mul!(QX, Q.S, X, -2.0, 1.0)
    mul!(QX, X, Q.T, -2.0, 1.0)
end


CUDA.@fastmath @inline function update_zxw1_kernel!(dx::CuDeviceVector{Float64},
                                                    w_bar::CuDeviceVector{Float64},
                                                    w::CuDeviceVector{Float64},
                                                    z_bar::CuDeviceVector{Float64},
                                                    x_bar::CuDeviceVector{Float64},
                                                    x_hat::CuDeviceVector{Float64},   
                                                    last_x::CuDeviceVector{Float64},
                                                    x::CuDeviceVector{Float64},
                                                    Qw::CuDeviceVector{Float64},
                                                    ATy::CuDeviceVector{Float64},
                                                    c::CuDeviceVector{Float64},
                                                    tempv::CuDeviceVector{Float64},
                                                    l::CuDeviceVector{Float64},
                                                    u::CuDeviceVector{Float64},
                                                    sigma::Float64,
                                                    fact1::Float64,
                                                    fact2::Float64,
                                                    Halpern_fact1::Float64,
                                                    Halpern_fact2::Float64,
                                                    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
    end
    return
end

function update_zxw1_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    QAP_Qmap!(ws.w, ws.Qw, ws.temp1, qp.Q)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_kernel!(dy::CuDeviceVector{Float64},
                        y_bar::CuDeviceVector{Float64},
                        y::CuDeviceVector{Float64},
                        last_y::CuDeviceVector{Float64},
                        s::CuDeviceVector{Float64},
                        AL::CuDeviceVector{Float64},
                        AU::CuDeviceVector{Float64},
                        Ax::CuDeviceVector{Float64},
                        fact1::Float64,
                        fact2::Float64,
                        Halpern_fact1::Float64,
                        Halpern_fact2::Float64,
                        m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        s[i] = Ax[i] - fact1 * y[i]
        y_bar[i] = s[i] < AL[i] ? (AL[i] - s[i]) : (s[i] > AU[i] ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function update_y_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    QAP_Qmap!(ws.w_bar, ws.Qw_bar, ws.temp1, qp.Q)
    axpby_gpu!(1.0, ws.tempv, -ws.sigma, ws.Qw_bar, ws.tempv, ws.n)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
end

function update_w2_kenel!(dw::CuDeviceVector{Float64},
                        ATdy::CuDeviceVector{Float64},
                        w::CuDeviceVector{Float64},
                        w_bar::CuDeviceVector{Float64},
                        last_w::CuDeviceVector{Float64},
                        last_ATy::CuDeviceVector{Float64},
                        ATy::CuDeviceVector{Float64},
                        ATy_bar::CuDeviceVector{Float64},
                        fact::Float64,
                        Halpern_fact1::Float64,
                        Halpern_fact2::Float64,
                        n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = w_bar[i] + fact * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w2_kenel!(ws.dw, ws.ATdy, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q), Halpern_fact1, Halpern_fact2, ws.n)
end

## kernels for compute residuals

function compute_Rd_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qx[i] + c[i] - ATy[i] - z[i]
        scale_fact = col_norm[i] * c_scale
        Rd[i] *= scale_fact
        Qx[i] *= scale_fact
        ATy[i] *= scale_fact
    end
    return
end

function compute_Rd_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    if ws.m > 0
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
end

function compute_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= m
        Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : (Ax[i] > AU[i] ? (AU[i] - Ax[i]) : 0.0)
        scale_fact = row_norm[i] * b_scale
        Rp[i] *= scale_fact
        Ax[i] *= scale_fact
    end
    return
end

function compute_Rp_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
end

function compute_err_lu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0)
        dx[i] *= b_scale / col_norm[i]
    end
    return
end

function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64},
    b::Float64, y::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float64, x::CuArray{Float64},
    b::Float64, y::CuArray{Float64},
    z::CuArray{Float64}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

## function for update sigma

@inline function f_dev(x, a, b, c, d)
    return a * x + b / x + c * x^2 / (1 + d * x)
end

function golden(
    a_p::Float64, b_p::Float64, c_p::Float64, d_p::Float64;
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200
)
    # golden ratio constant
    φ = (sqrt(5.0) - 1.0) / 2.0
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

    x_sol = 0.5 * (a + b)
    return x_sol
end