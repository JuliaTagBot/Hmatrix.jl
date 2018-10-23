using PyPlot
using HCubature
using SpecialFunctions
using Polynomials
using PyCall
using ProgressMeter
using DelimitedFiles
using ToeplitzMatrices
using LinearAlgebra
using IterativeSolvers
using JLD2
using StatsFuns
using Statistics
using FFTW
using Printf
using FileIO
using Interpolations


@pyimport numpy
@pyimport scipy.signal as ss
@pyimport scipy.sparse.linalg as ssl
@pyimport scipy.stats as sstats
@pyimport scipy.special as ssp


function pygmres(A, x, M=nothing; x0=nothing)
    N = length(x)
    lo = ssl.LinearOperator((N, N), matvec=A)
    if M==nothing
        Mop = ssl.LinearOperator((N, N), matvec=x->x)
    else
        Mop = ssl.LinearOperator((N, N), matvec=x->M\x)
    end
    # if x0==nothing
    #     x0 = zeros(size(x))
    # end
    y = ssl.gmres(lo, x, M = Mop, x0=x0)[1]
    return y
end

function pycg(A, x, M=nothing)
    N = length(x)
    lo = ssl.LinearOperator((N, N), matvec=A)
    if M==nothing
        Mop = ssl.LinearOperator((N, N), matvec=x->x)
    else
        Mop = ssl.LinearOperator((N, N), matvec=x->M\x)
    end
    y = ssl.cg(lo, x, M = Mop)[1]
    return y
end

function pygmres_mat_callback(rk)
    global cnt
    cnt += 1
end

function pygmres_mat(A, x, M)
    global cnt
    cnt = 0
    N = length(x)
    lo = ssl.LinearOperator((N, N), matvec=x->A*x)
    Mop = ssl.LinearOperator((N, N), matvec=x->M\x)
    y = ssl.gmres(lo, x, callback=PyCall.jlfun2pyfun(pygmres_mat_callback),M = Mop)[1]
    return y, cnt
end

function conv_wrapper(M, v)
    return numpy.convolve(reverse(M), v, "same")
end

# https://en.wikipedia.org/wiki/Smoothstep
function rhor(x, r)
    x = abs(x/r)
    if x<1
        return 1-(70x^9-315x^8+540x^7-420x^6+126x^5)
    else
        return 0
    end
end

# https://arxiv.org/pdf/1311.7691.pdf
function C_n_s( n, s)
    alpha = 2s
    return alpha*2^(alpha-1)*gamma((alpha+n)/2)/pi^(n/2)/gamma((2-alpha)/2)
end



function compute_M(Lw, n, r, m, gW)
    N=2m+1
    rho = x->rhor(x, r)

    y = LinRange(-Lw, Lw, N)
    h = y[2]-y[1]
    w = h * ones(N)
    w[1] = h/2; w[end] = h/2

    M = zeros(N)
    for i = -m:m
        idx = i + m + 1
        if i==0
            continue
        end
        M[idx] += n(y[idx])*w[idx]
    end

    M[m+1] = -sum(M)

    for i = -m:m
        idx = i + m + 1
        if i==0
            continue
        end
        M[m+2] -= rho(y[idx])*y[idx]*n(y[idx])*w[idx]/h
        M[m] += rho(y[idx])*y[idx]*n(y[idx])*w[idx]/h
    end


    for i = -m:m
        idx = i + m + 1
        if i==0
            continue
        end
        M[m+2] -= 1/2*rho(y[idx])*n(y[idx])*y[idx]^2*w[idx]/h^2
        M[m] -= 1/2*rho(y[idx])*n(y[idx])*y[idx]^2*w[idx]/h^2
        M[m+1] += rho(y[idx])*n(y[idx])*y[idx]^2*w[idx]/h^2
    end


    c = hquadrature(y->rho(y)*n(y)*y^2, 0, r)[1] + hquadrature(y->rho(y)*n(-y)*y^2, -r, 0)[1]
    # c = tanh_sinh_integral(y->rho(y)*n(y)*y^2, 0, r) +
    #     tanh_sinh_integral(y->rho(y)*n(-y)*y^2, -r, 0) 
    # println(c)
    M[m+2] += 1/2*c/h^2
    M[m] += 1/2*c/h^2
    M[m+1] -= c/h^2

    M[m+1] -= gW

    return M
end

function compute_K(M, m)
    N = Int((length(M)-1)/2)
    return Toeplitz( M[N+1:-1:N-2m+1], M[N+1:N+1+2m] )
end

function compute_fW(L, m, fWx)
    x = LinRange(-L, L, 2m+1)
    y = fWx.(x)
    return y
end

function compute_fW(a, b, m, fWx)
    x = LinRange(a,b,2m+1)
    y = fWx.(x)
end

function compute_hW(M, u, N, m)
    # assert(length(u)==2m+2N+1)
    f = conv_wrapper(M, u)
    return f[N+1:N+1+2m]
end


function opL(Lw, N, L, m, n, r, gW, fWx, un; A, M)
    if A == nothing
        M = compute_M(Lw, n, r, N, gW)
        A = compute_K(M, m)
    end
    fW = compute_fW(L, m, fWx)
    gW = compute_hW(M, un, N, m)
    return A, fW+gW, M
end

function opD1(L, m, ul, ur; A)
    h = L/m
    if A==nothing
        dl = -ones(2m)/(2h)
        du = ones(2m)/(2h)
        d = zeros(2m+1)
        A = Tridiagonal(dl, d, du)
    end
    f = zeros(2m+1)
    f[1] = -ul/(2h)
    f[end] = ur/(2h)
    return A, f
end

function opD2(L, m, ul, ur; A)
    h = L/m

    if A==nothing
        dl = ones(2m)/h^2
        du = ones(2m)/h^2
        d = -2*ones(2m+1)/h^2
        A = Tridiagonal(dl, d, du)
    end
    f = zeros(2m+1)
    f[1] = ul/h^2
    f[end] = ur/h^2
    return A, f
end

# gW : Float64
# fWx : function of (x,t)
# r, b, sigma, lambda : function of (x, t)
# rW: window length
function explicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, r, b, sigma, lambda, u_l, u_r, u0; nonlocal = true)
    x = LinRange(-L, L, 2m+1)
    h = x[2]-x[1]
    dt = T/Tn
    U = zeros(2m+1, Tn+1)
    U[:,1] = u0.(x)
    M = nothing
    D1 = nothing
    D2 = nothing
    Ds = nothing
    for i = 1:Tn
        t = i*dt
        Rt = r.(x, t)
        Bt = b.(x, t)
        St = sigma.(x, t)
        Lt = lambda.(x, t)

        ul = u_l(-L-h, t)
        ur = u_r(L+h, t)
        un = zeros(2N+2m+1)
        un[1:N] = u_l.(-L-N*h:h:-L-h/2, t)
        un[2m+N+2:2N+2m+1] = u_r.(L+h:h:L+N*h+h/2, t)
        D1, f1 = opD1(L, m, ul, ur; A=D1)
        D2, f2 = opD2(L, m, ul, ur; A=D2)
        if nonlocal
            Ds, fs, M = opL(Lw, N, L, m, n, rW, gW, x->fWx(x, t), un; A=Ds, M=M)
        else
            Ds = UniformScaling(0.0)
            fs = zeros(length(f1))
        end


        # println(D1)
        # println(D2)
        # println(Ds)

        U[:,i+1] = U[:,i] + dt*(
            Rt.*U[:,i] +
            Bt.*(D1*U[:,i]+f1) +
            St.*(D2*U[:,i]+f2) +
            Lt.*(Ds*U[:,i]+fs))
    end
    return U
end

function implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, r, b, sigma, lambda, u_l, u_r, u0; matrix=false, nonlocal=true)
    x = LinRange(-L, L, 2m+1)
    h = x[2]-x[1]
    dt = T/Tn
    U = zeros(2m+1, Tn+1)
    U[:,1] = u0.(x)
    M = nothing
    D1 = nothing
    D2 = nothing
    Ds = nothing
    @showprogress 1 "Implicit..."  for i = 1:Tn
        t = i*dt
        Rt = r.(x, t)
        Bt = b.(x, t)
        St = sigma.(x, t)
        Lt = lambda.(x, t)

        ul = u_l(-L-h, t)
        ur = u_r(L+h, t)
        un = zeros(2N+2m+1)
        un[1:N] = u_l.(-L-N*h:h:-L-h/2, t)
        un[2m+N+2:2N+2m+1] = u_r.(L+h:h:L+N*h+h/2, t)
        D1, f1 = opD1(L, m, ul, ur; A=D1)
        D2, f2 = opD2(L, m, ul, ur; A=D2)
        if nonlocal
            Ds, fs, M = opL(Lw, N, L, m, n, rW, gW, x->fWx(x, t), un; A=Ds, M=M)
        else
            Ds = UniformScaling(0.0)
            fs = zeros(length(f1))
        end
        g = dt*(Bt.*f1 + St.*f2 + Lt.*fs)+U[:,i]
        opK = x->x - dt*Rt.*x - dt*Bt.*(D1*x) - dt*St.*(D2*x) - dt*Lt.*(Ds*x)
        U[:,i+1] = pygmres(opK, g)
        if matrix && i==1
            D1 = Array{Float64}(D1)
            D2 = Array{Float64}(D2)
            Ds = Array{Float64}(Ds)
            @save "data.jld" dt Rt Bt St Lt D1 D2 Ds
            return
        end
    end

    return U
end

function example_opL()
    m = 100 # resolution: [a,b] is split into 2m+1 intervals
    s = 0.9; alpha=2s
    Lw = 2.0 # near-field range
    n = x->1/abs(x)^(1+2s) # kernel
    r = 0.3 # window function
    a = -1.0
    b = 1.0
    N = 2m # must be consistent N = m*Int(Lw/L)
    gW = 1/(s*Lw^(2s)) # far-field contribution
    fWx = x->0 # int_{|x+y|>=L_W} u(x+y)n(y) dy
    u = zeros(2(N+m)+1) # must be consistent

    c0 = C_n_s(1, s)
    c1 = 2^(-alpha)*gamma(1/2)/gamma(1+alpha/2)/gamma((1+alpha)/2)

    M = compute_M(Lw, n, r, N, gW)
    K = compute_K(M, m)
    fW = compute_fW(-1., 1., m, fWx)
    hW = compute_hW(M, u, N, m)
    # @show M
    
    K *= -c0
    fW *= -c0
    hW *= -c0

    rhs = ones(2m+1)
    # sol = K\(rhs - fW - hW)
    sol = pycg(x->K*x, rhs-fW-hW)
    x = LinRange(-1.0,1.0,2m+1)
    plot(x, sol/c1, "*-", label="numerical")
    plot(x,(1 .- x.^2).^s, label="exact")
    legend()
end

function example_opL_rate()

    function do_experiment(m)
        s = 0.1; alpha=2s
        Lw = 2.0 # near-field range
        n = x->1/abs(x)^(1+2s) # kernel
        r = 0.3 # window function
        a = -1.0
        b = 1.0
        L = (b-a)/2.0 # computational domain
        N = m*Int(Lw/L) # must be consistent N = m*Int(Lw/L)
        gW = 1/(s*Lw^(2s)) # far-field contribution
        fWx = x->0 # int_{|x+y|>=L_W} u(x+y)n(y) dy
        u = zeros(2(N+m)+1) # must be consistent

        c0 = C_n_s(1, s)
        c1 = 2^(-alpha)*gamma(1/2)/gamma(1+alpha/2)/gamma((1+alpha)/2)

        M = compute_M(Lw, n, r, N, gW)
        K = compute_K(M, m)
        fW = compute_fW(-1., 1., m, fWx)
        hW = compute_hW(M, u, N, m)
        
        K *= -c0
        fW *= -c0
        hW *= -c0

        rhs = ones(2m+1)
        # sol = K\(rhs - fW - hW)
        sol = pycg(x->K*x, rhs-fW-hW)/c1
        x = LinRange(-1.0,1.0,2m+1)
        return sqrt(mean(abs.(sol- (1 .- x.^2).^s).^2))
    end

    MM = [50,100,150,200,400,800]
    E = zeros(length(MM))
    for i = 1:length(MM)
        E[i] = do_experiment(MM[i])
        @show i, E[i]
    end
    r = round(polyfit(log.(MM), log.(E), 1)[1],digits = 2)
    loglog(MM, E, "*-", label="rate=$r")
    legend()
end
#=
function bseq()
    r = 0.1
    s = 0.4

    L = 4.0
    m = 100
    Lw = 12.0
    N = m*Int(Lw/L)
    n = x->0
    rW = 0.5
    gW = 0.0
    fWx = (x,t)->0
    T = 0.25
    Tn = 100
    rfun = (x,t)-> -r
    bfun = (x,t)->r-s^2/2
    sigma = (x,t)->1/2*s^2
    lambda = (x,t)->0
    u_l = (x,t)-> 0
    u_r = (x,t)-> exp(x)
    u0 = x -> max(exp(x)-10, 0)
    U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0)
    t = LinRange(0, T, Tn+1)
    x = LinRange(-L, L, 2m+1)
    # mesh(t, exp.(x), U)
    plot(exp.(x), U[:,end],".")
    plot(exp.(x), exact_sol.(x, 10, r, s, T, T))
    xlim(0,40)
    xlabel("Stock Price")
    ylabel("Option Price")
    grid()
    legend()
end

function cgmy()
    r = 0.1
    s = 0.4
    C = 1.0
    G = 1.5
    M = 1.5
    Y = 0.5

    L = 4.0
    m = 100
    Lw = 8.0
    N = m*Int(Lw/L)
    function n(x)
        if x<0
            return C*exp(-G*abs(x))/abs(x)^(1+Y)
        else
            return C*exp(-M*abs(x))/abs(x)^(1+Y)
        end
    end
    rW = 0.5
    gW = 0.0
    fWx = (x,t)->0
    T = 0.25
    Tn = 100
    rfun = (x,t)-> -r
    bfun = (x,t)->r-s^2/2
    sigma = (x,t)->1/2*s^2
    lambda = (x,t)->1.0
    u_l = (x,t)-> 0
    u_r = (x,t)-> exp(x)
    u0 = x -> max(exp(x)-10, 0)
    U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0)
    t = LinRange(0, T, Tn+1)
    x = LinRange(-L, L, 2m+1)
    # mesh(t, exp.(x), U)
    plot(exp.(x), U[:,end],".")
    plot(exp.(x), exact_sol.(x, 10, r, s, T, T), label="Black-Sholes Equation")
    xlim(0,40)
    xlabel("Stock Price")
    ylabel("Option Price")
    grid()
    legend()
end

function cgmy_matrix(m)
    r = 0.1
    s = 0.4
    C = 1.0
    G = 0.0
    M = 0.0
    Y = 1.1

    L = 4.0
    Lw = 8.0
    N = m*Int(Lw/L)
    function n(x)
        if x<0
            return C*exp(-G*abs(x))/abs(x)^(1+Y)
        else
            return C*exp(-M*abs(x))/abs(x)^(1+Y)
        end
    end
    rW = 0.5
    gW = 0.0
    fWx = (x,t)->0
    T = 0.25
    Tn = 100
    rfun = (x,t)-> -r
    bfun = (x,t)->r-s^2/2
    sigma = (x,t)->1/2*s^2
    lambda = (x,t)->1.0

    lambda = (x,t)->1.0
    u_l = (x,t)-> 0
    u_r = (x,t)-> exp(x)
    u0 = x -> max(exp(x)-10, 0)
    implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0; matrix=true)
    @load "data.jld" dt Rt Bt St Lt D1 D2 Ds

    R = diagm(0=>Rt)
    B = diagm(0=>Bt)
    S = diagm(0=>St)
    L = diagm(0=>Lt);

    # A = one(R)-dt*R - dt*B*D1 - dt*S*D2 - dt*L*Ds;
    M0 = one(R)-dt*R - dt*B*D1 - dt*S*D2
    # M = - M0\( dt*L )
    A = M0 - dt * L * Ds
    # A = M+dt*L*Ds
    sol, cnt = pygmres_mat(A, rand(length(Rt)), (M0))
    println(cnt)
    return A, D2, Ds
end

function cgmy2()

    TT = [20, 40, 80, 160, 320, 640]
    m = 1000
    U0 = zeros(m*2+1, 6)
    for (kk, Tn) in enumerate(TT)
        r = 0.1
        s = 0.4
        C = 1.0
        G = 1.5
        M = 1.5
        Y = 0.5

        L = 4.0

        Lw = 8.0
        N = m*Int(Lw/L)
        function n(x)
            if x<0
                return C*exp(-G*abs(x))/abs(x)^(1+Y)
            else
                return C*exp(-M*abs(x))/abs(x)^(1+Y)
            end
        end
        rW = 0.5
        gW = 0.0
        fWx = (x,t)->0
        T = 0.25
        rfun = (x,t)-> -r
        bfun = (x,t)->r-s^2/2
        sigma = (x,t)->1/2*s^2
        lambda = (x,t)->1.0
        u_l = (x,t)-> 0
        u_r = (x,t)-> exp(x)
        u0 = x -> max(exp(x)-10, 0)
        U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0)
        t = LinRange(0, T, Tn+1)
        x = LinRange(-L, L, 2m+1)
        # mesh(t, exp.(x), U)
        U0[:, kk] = U[:,end]
    end
    E = zeros(5)
    for k = 1:5
        E[k] = norm(U0[:,k]-U0[:,6])
    end
    loglog(TT[1:5], E,"*-")

end

function cgmy_dist()
    r = 0.1
    s = 0.0
    C = 1.
    G = 1.5
    M = 1.5
    Y = 0.5

    L = 4.0
    m = 500
    Lw = 8.0
    N = m*Int(Lw/L)
    function n(x)
        if x<0
            return C*exp(-G*abs(x))/abs(x)^(1+Y)
        else
            return C*exp(-M*abs(x))/abs(x)^(1+Y)
        end
    end
    rW = 0.5
    gW = 0.0
    fWx = (x,t)->0
    T = 0.25
    Tn = 500
    nu = C*gamma(-Y)*((M-1)^Y -M^Y + (G+1)^Y - G^Y)
    r = r-nu
    rfun = (x,t)-> -r
    # bfun = (x,t)->r-s^2/2
    bfun = (x,t)->0.0
    sigma = (x,t)->1/2*s^2
    lambda = (x,t)->1.0
    u_l = (x,t)-> 0
    u_r = (x,t)-> 0

    function u0(x)
        if abs(x)<1e-5
            return 1.0
        else
            return 0.0
        end
    end

    U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0)
    t = LinRange(0, T, Tn+1)
    x = LinRange(-L, L, 2m+1)
    # mesh(t, exp.(x), U)
    plot(x, U[:,100]/sum(U[1:end-1,100]*(L/m)), label="t=0.05")
    plot(x, U[:,300]/sum(U[1:end-1,250]*(L/m)), label="t=0.15")
    plot(x, U[:,end]/sum(U[1:end-1,end]*(L/m)), label="t=0.25")
    xlabel("x")
    ylabel("PDF")
    legend()
    grid()

end

function exact_sol(x, E, r, sigma, T, t)
    N(v) = sstats.norm[:cdf](v)
    S = exp(x)
    d1 = (log(S/E)/log(exp(1))+(r+1/2*sigma^2)*(t))/(sigma*sqrt(t))
    d2 = d1 - sigma*sqrt(t)
    return S*N(d1)-E*exp(-r*(t))*N(d2)
end

function kay()
    kappa = 0.0117
    s = 0.013
    theta = 0.0422
    Lambda0 = 0.011
    Lambda1 = 0.1

    L = 0.1
    m = 500
    Lw = 0.2
    N = m*Int(Lw/L)
    function n(x)
        if x<0
            return 0.0
        else
            return x
        end
    end
    rW = 0.5
    gW = 0.0
    fWx = (x,t)->0
    T = 0.25
    Tn = 500

    rfun = (x,t)-> 0.0
    bfun = (x,t)-> kappa*(theta-x)
    sigma = (x,t)-> 1/2*s^2*max(x,0)
    lambda = (x,t)-> Lambda0 + Lambda1*x
    u_l = (x,t)-> 0
    u_r = (x,t)-> 0

    function u0(x)
        if abs(x-0.0422)<L/m/2
            return 1.0
        else
            return 0.0
        end
    end

    U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0)
    t = LinRange(0, T, Tn+1)
    x = LinRange(-L, L, 2m+1)
    # mesh(t, exp.(x), U)
    plot(x, U[:,100]/sum(U[1:end-1,100]*(L/m)), label="t=0.05")
    plot(x, U[:,300]/sum(U[1:end-1,250]*(L/m)), label="t=0.15")
    plot(x, U[:,end]/sum(U[1:end-1,end]*(L/m)), label="t=0.25")
    xlabel("x")
    ylabel("PDF")
    legend()
    grid()
end

function cir_xt(y, x,a,b,s,T)
    d = 4a/s^2
    l = s^2/(4b)*(1-exp(-b*T))
    return nchisqpdf( d, x/l, y/l)/l
end

function cir()
    a = 0.07
    b = 0.25
    a = 0
    b = 0
    s = 0.3

    L = 0.5
    m = 300
    Lw = L
    N = m*Int(Lw/L)
    function n(x)
        0.0
    end
    rW = 0.05
    gW = 0.0
    fWx = (x,t)->0
    T = 1.0
    Tn = 500

    rfun = (x,t)-> 0
    bfun = (x,t)-> -b*(x+L)+a + s^2
    sigma = (x,t)-> 1/2*s^2*(x+L)
    lambda = (x,t)-> 0.0
    u_l = (x,t)-> 0.0
    u_r = (x,t)-> 0.0

    function u0(x)
        # if abs(x + L-0.02)<L/2m
        #     return 1.0/(L/m)
        # else
        #     return 0.0
        # end
        return cir_xt.(x+L, 0.02,a,b,s,0.0001)
    end

    U = implicit(L, m, Lw, N, n, rW, gW, fWx, T, Tn, rfun, bfun, sigma, lambda, u_l, u_r, u0; nonlocal=false)
    t = LinRange(0, T, Tn+1)
    x = LinRange(-L, L, 2m+1)
    # mesh(t, exp.(x), U)
    # plot(x .+ L, U[:,100],"+--", label="t=$(T/5)")
    # plot(x .+ L, U[:,200],"+--", label="t=$(T/5*2)")
    plot(x .+ L, U[:,300],"+--", label="t=$(T/5*3)")
    plot(x .+ L, U[:,400],"+--", label="t=$(T/5*4)")
    plot(x .+ L, U[:,end],"+--", label="t=$T")
    y = cir_xt.(x .+ 0.5, 0.02,a,b,s,T)
    plot(x .+ 0.5, y, label="Exact t=1.0")
    xlim([0,0.2])
    xlabel("exp(x)")
    ylabel("PDF")
    legend()
    grid()
    # mesh(U)
    return U
end
=#
