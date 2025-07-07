## normal z x w1 y w2 kernels
CUDA.@fastmath @inline function update_zxw1_kernel!(dx::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64}, z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
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

function update_zxw1_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_kernel!(dy::CuDeviceVector{Float64}, y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64}, last_y::CuDeviceVector{Float64}, s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64}, AU::CuDeviceVector{Float64}, Ax::CuDeviceVector{Float64},
    fact1::Float64, fact2::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        s[i] = Ax[i] - fact1 * y[i]
        y_bar[i] = (s[i] < AL[i]) ? (AL[i] - s[i]) : (s[i] > AU[i] ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1.0, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    axpby_gpu!(1.0, ws.tempv, -ws.sigma, ws.Qw_bar, ws.tempv, ws.n)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

function update_w2_kenel!(dw::CuDeviceVector{Float64}, ATdy::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    last_ATy::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, ATy_bar::CuDeviceVector{Float64},
    fact::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
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

## normal z x w1 y w2 kernels (customized)

CUDA.@fastmath @inline function cust_update_zxw1_kernel!(dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w[colValQ[k]]
        end
        Qw[i] = qr1
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
    end
    return
end

function cust_update_zxw1_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_zxw1_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function cust_update_w2_kernel!(dw::CuDeviceVector{Float64}, ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64}, last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, ATy_bar::CuDeviceVector{Float64},
    fact::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_bar[colValAT[k]]
        end
        ATy_bar[i] = sAT

        w_bar[i] = w_bar[i] + fact * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function cust_update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w2_kernel!(ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar,
        ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q), Halpern_fact1, Halpern_fact2, ws.n)
end

## zxw1, w2 kernels for diagonal Q

CUDA.@fastmath @inline function update_zxw1_diagQ_kernel!(dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
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

        w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
    end
    return
end

function update_zxw1_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w, 0, ws.Qw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw1_diagQ_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

# special case of Q = 0

CUDA.@fastmath @inline function update_zx_kernel!(dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        tmp = ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

function update_zx_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zx_kernel!(ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

# special case of Q = 0

function update_y_noQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

function update_w2_diagQ_kenel!(dw::CuDeviceVector{Float64}, ATdy::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64}, last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, ATy_bar::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64}, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        w_bar[i] = w_bar[i] + fact[i] * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function update_w2_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w2_diagQ_kenel!(ws.dw, ws.ATdy, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.fact, Halpern_fact1, Halpern_fact2, ws.n)
end

## zxw1, w2 kernels for diagonal Q (customized)

CUDA.@fastmath @inline function cust_update_zxw1_diagQ_kernel!(dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w[colValQ[k]]
        end
        Qw[i] = qr1
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp
        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i]
        w_bar[i] = fact1[i] * w[i] + fact2[i] * x_hat[i]
    end
    return
end

function cust_update_zxw1_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_zxw1_diagQ_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

# special case of Q = 0 (customized)

CUDA.@fastmath @inline function cust_update_zx_kernel!(dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        v = 0.0
        @inbounds for k in startAT:stopAT
            v += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = v
        tmp = ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp
        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

function cust_update_zx_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_zx_kernel!(ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.l, ws.u, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

function cust_compute_r2_kernel!(rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    sigma::Float64, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        tempv[i] = x_hat[i] + sigma * (Qw[i] - qr1)
    end
    return
end

CUDA.@fastmath @inline function cust_update_y_kernel!(dy::CuDeviceVector{Float64},
    rowPtrA::CuDeviceVector{Int32}, colValA::CuDeviceVector{Int32}, nzValA::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64}, y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64}, s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64}, AU::CuDeviceVector{Float64},
    fact1::Float64, fact2::Float64,
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        startA = rowPtrA[i]
        stopA = rowPtrA[i+1] - 1
        Ai = 0.0
        @inbounds for k in startA:stopA
            Ai += nzValA[k] * tempv[colValA[k]]
        end
        s[i] = Ai - fact1 * y[i]
        y_bar[i] = s[i] < (AL[i]) ? (AL[i] - s[i]) : (s[i] > (AU[i]) ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
    end
    return
end

function cust_update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_compute_r2_kernel!(ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.Qw, ws.sigma, ws.x_hat, ws.tempv, ws.n)
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.tempv, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.lambda_max_A * ws.sigma, 1 / (ws.lambda_max_A * ws.sigma), Halpern_fact1, Halpern_fact2, ws.m)
    end
end

# special case of Q = 0 (customized)

function cust_update_y_noQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.x_hat, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.lambda_max_A * ws.sigma, 1 / (ws.lambda_max_A * ws.sigma), Halpern_fact1, Halpern_fact2, ws.m)
    end
end

CUDA.@fastmath @inline function cust_update_w2_diagQ_kernel!(dw::CuDeviceVector{Float64},
    ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64}, w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64}, last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, ATy_bar::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64}, Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y_bar[colValAT[k]]
        end
        ATy_bar[i] = sAT

        w_bar[i] = w_bar[i] + fact[i] * (ATy_bar[i] - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar[i] - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar[i] - ATy[i])
    end
    return
end

function cust_update_w2_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w2_diagQ_kernel!(ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y_bar,
        ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy, ws.ATy_bar, ws.fact, Halpern_fact1, Halpern_fact2, ws.n)
end

## w x y normal update kernels, when without C

CUDA.@fastmath @inline function update_w_noC_kernel!(dw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64,
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function update_x_noC_kernel!(dx::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64}, last_Qw::CuDeviceVector{Float64},
    sigma::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

function update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_x_noC_kernel!(ws.dx, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

## w x y normal update kernels, when without C (customized)

CUDA.@fastmath @inline function cust_update_w_noC_kernel!(dw::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64,
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function cust_update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function cust_update_x_noC_kernel!(dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    last_Qw::CuDeviceVector{Float64},
    sigma::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        Qw_bar[i] = qr1

        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        dx[i] = x_bar[i] - x[i]
        x_hat[i] = 2 * x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

function cust_update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_x_noC_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end

function cust_update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.x_hat, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

# Update w, x, y kernels for no C case with diagonal Q

CUDA.@fastmath @inline function update_w_noC_diagQ_kernel!(dw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_diagQ_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

# Update w, x, y kernels for no C case with diagonal Q (customized)

CUDA.@fastmath @inline function cust_update_w_noC_diagQ_kernel!(dw::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function cust_update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_diagQ_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

## kernels used to update sigma

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



# Golden-section search for minimizing 
# f(x) = a*x + b/x + x^2 * dot(c, (I + x*Q) \ d)
# GPU‑enabled golden‐section search for 
# f(x) = a*x + b/x + x^2 * dot(c, d ./ (1 + x*Q))
function golden_Q_diag(a::Float64, b::Float64, Q::CuArray{Float64}, c::CuArray{Float64}, d::CuArray{Float64}, tempv::CuArray{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using GPU operations, reusing tempv
    function f_gpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * CUDA.dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_gpu(x1)
    f2 = f_gpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_gpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_gpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end

#############################
# CUDA kernel to update all four factors in one pass
#############################
function update_Q_factors_kernel!(
    fact2::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64},
    fact1::CuDeviceVector{Float64},
    fact_M::CuDeviceVector{Float64},
    diag_Q::CuDeviceVector{Float64},
    sigma::Float64,
    s2::Float64,
    N::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

#############################
# High-level wrapper to launch the above kernel
#############################
function update_Q_factors_gpu!(
    fact2::CuVector{Float64},
    fact::CuVector{Float64},
    fact1::CuVector{Float64},
    fact_M::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    threads = 256
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks update_Q_factors_kernel!(
        fact2, fact, fact1, fact_M,
        diag_Q, sigma, s2, N
    )
    return
end

## kernels to compute residuals

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
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
end

function compute_Rd_noC_kernel!(ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qx[i] + c[i] - ATy[i]
        scale_fact = col_norm[i] * c_scale
        Rd[i] *= scale_fact
        Qx[i] *= scale_fact
        ATy[i] *= scale_fact
    end
    return
end

function compute_Rd_noC_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_noC_kernel!(ws.ATdy, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
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
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
    end
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