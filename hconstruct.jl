include("hmat.jl")
using Random

############### Compression ##################
function construct_hmat(A, Nleaf, Erank, Rrank, MaxBlock=64)
    function helper(H, A)
        H.m, H.n = size(A,1), size(A, 2)
        if size(A,1)>MaxBlock
            k = Rrank
        else
            U,S,V = svd(A)
            k = rank_truncate(S,Erank)
        end
        if k < Rrank
            H.is_rkmatrix = true
            H.A = U[:,1:k]
            H.B = V[:,1:k] * diagm(0=>S[1:k])
        elseif size(A,1)<=Nleaf
            H.is_fullmatrix = true
            H.C = copy(A)
        else
            H.is_hmat = true
            H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
            n = Int(round.(size(A,1)/2))
            m = Int(round.(size(A,2)/2))
            @views begin
                helper(H.children[1,1], A[1:n, 1:m])
                helper(H.children[1,2], A[1:n, m+1:end])
                helper(H.children[2,1], A[n+1:end, 1:m])
                helper(H.children[2,2], A[n+1:end, m+1:end])
            end
        end
        # consistency(H)
    end

    H = Hmat()
    Ac = copy(A)
    helper(H, Ac)
    return H
end

############### Construction from Kernel Functions ##################
function admissible1(s1, e1, s2, e2)
    if s1<=s2 && s2<e1 
        return true
    elseif s1>=s2 && s1<e2
        return true
    else
        return false
    end
end

function admissible2(s1, e1, s2, e2)
    p = e1-s1
    if s1>=e2 || s2>=e1
        if s1>e2
            d = s1-e2
        else
            d = s2-e1
        end
        return d>p
        # return true
    else
        return false
    end
end

function construct1D(kerfun, N1, N2, Nleaf, Rrank, MaxBlock)
    function helper(H::Hmat, s1, e1, s2, e2)
        # println("$s1, $e1, $s2 ,$e2")
        H.m = e1-s1+1
        H.n = e2-s2+1
        if H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
            H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
            m1 = s1 + Int(round((e1-s1)/2)) - 1
            m2 = s2 + Int(round((e2-s2)/2)) - 1
            helper(H.children[1,1], s1, m1, s2, m2)
            helper(H.children[1,2], s1, m1, m2+1, e2)
            helper(H.children[2,1], m1+1, e1, s2, m2)
            helper(H.children[2,2], m1+1, e1, m2+1, e2)
        elseif !admissible1(s1, e1, s2, e2)
            J = zeros(H.m, H.n)
            for i = 1:H.m
                for j = 1:H.n
                    J[i,j] = kerfun(s1-1+i, s2-1+j)
                end
            end
            U,S,V = psvd(J)
            # k = length(S)
            k = rank_truncate(S,1e-6)
            # println("$k, $(H.m), $(H.n)")
            if k < Rrank
                H.is_rkmatrix = true
                H.A = U[:,1:k]
                H.B = V[:,1:k] * diagm(0=>S[1:k])
            elseif H.m > Nleaf && H.n > Nleaf
                H.is_hmat = true
                H.children = Array{Hmat}([Hmat() Hmat()
                                        Hmat() Hmat()])
                m1 = s1 + Int(round((e1-s1)/2)) - 1
                m2 = s2 + Int(round((e2-s2)/2)) - 1
                helper(H.children[1,1], s1, m1, s2, m2)
                helper(H.children[1,2], s1, m1, m2+1, e2)
                helper(H.children[2,1], m1+1, e1, s2, m2)
                helper(H.children[2,2], m1+1, e1, m2+1, e2)
            else
                H.is_fullmatrix = true
                H.C = J
            end
        else
            if H.m > Nleaf && H.n > Nleaf
                H.is_hmat = true
                H.children = Array{Hmat}([Hmat() Hmat()
                                        Hmat() Hmat()])
                m1 = s1 + Int(round((e1-s1)/2)) - 1
                m2 = s2 + Int(round((e2-s2)/2)) - 1
                helper(H.children[1,1], s1, m1, s2, m2)
                helper(H.children[1,2], s1, m1, m2+1, e2)
                helper(H.children[2,1], m1+1, e1, s2, m2)
                helper(H.children[2,2], m1+1, e1, m2+1, e2)
            else
                H.is_fullmatrix = true
                H.C = zeros(H.m, H.n)
                for i = 1:H.m
                    for j = 1:H.n
                        H.C[i,j] = kerfun(s1-1+i, s2-1+j)
                    end
                end
            end
        end     
    end
    H = Hmat()
    helper(H, N1, N2, N1, N2)
    return H
end

function kerf1(i, j)
    if i==j
        return 10
    else
        return 1/(i-j)^2
    end
end

function fast_construct_rk_mat(alpha, beta, x, y)
    xbar = mean(x)
    t0 = x .- xbar
    t = y .- xbar
    n = length(x)
    m = length(y)
    r = length(alpha)
    U = zeros(n, r)
    V = zeros(m, r)
    for i = 1:r
        U[:,i] = alpha[i].(t0)
        V[:,i] = beta[i].(t)
    end
    return U, V
end



function full_mat(f, x, y)
    n = length(x)
    m = length(y)
    U = zeros(n, m)
    for i = 1:n
        for j = 1:m
            U[i,j] = f(x[i], y[j])
        end
    end
    return U
end

function Merton_Kernel(eps, r)
    f = (x,y)->exp(-eps^2*(x-y)^2)
    alpha = []
    beta = []
    for i = 0:r-1
        push!(alpha, t->exp(-eps^2*t^2)*t^i*2^i*eps^(2i)/factorial(i))
        push!(beta, t->exp(-eps^2*t^2)*t^i)
    end
    return f, alpha, beta
end

function mk2d_1(t, s, m, n, eps)
    return (2*eps^2)^(m+n)/factorial(m)/factorial(n)*s^m*t^n*exp(-eps^2*(t^2+s^2))
end

function mk2d_2(t, s, m, n, eps)
    return t^n*s^m*exp(-eps^2*(t^2+s^2))
end

function Merton_Kernel2D(eps, r)
    f = (x,y)->exp(-eps^2*norm(x-y)^2)
    alpha = []
    beta = []
    for m = 0:r-1
        for n = 0:r-1
            push!(alpha, (s,t)->mk2d_1(t, s, m, n, eps))
            push!(beta, (s,t)->mk2d_2(t, s, m, n, eps))
        end
    end
    return f, alpha, beta
end

function Zero_Kernel(eps, r)
    f = (x,y)->0.0
    alpha = []
    beta = []
    for i = 0:r-1
        push!(alpha, t->0.0)
        push!(beta, t->0.0)
    end
    return f, alpha, beta
end

function test_r_low_rank_block()
    h = 0.01
    eps = 1
    x = collect(1:10)*h
    y = collect(11:21)*h
    f, alpha, beta = Merton_Kernel(eps, 5)
    U, V = fast_construct_rk_mat(alpha, beta, x, y)
    G = full_mat(f, x, y)
    println(G)
    println(U*V')
    println(U*V'-G)
    println(norm(G-U*V',2)/norm(G,2))
end



function construct1D_low_rank(f, alpha, beta, h, N1, N2, Nleaf, MaxBlock)
    function helper(H::Hmat, s1, e1, s2, e2)
        # println("$s1, $e1, $s2 ,$e2")
        H.m = e1-s1+1
        H.n = e2-s2+1
        if H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
            H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
            m1 = s1 + Int(round((e1-s1)/2)) - 1
            m2 = s2 + Int(round((e2-s2)/2)) - 1
            helper(H.children[1,1], s1, m1, s2, m2)
            helper(H.children[1,2], s1, m1, m2+1, e2)
            helper(H.children[2,1], m1+1, e1, s2, m2)
            helper(H.children[2,2], m1+1, e1, m2+1, e2)
        elseif admissible2(s1, e1, s2, e2)
            H.is_rkmatrix = true
            H.A, H.B = fast_construct_rk_mat(alpha, beta, collect(s1:e1)*h, collect(s2:e2)*h)
            # D = full_mat(f, collect(s1:e1)*h, collect(s2:e2)*h)
            # println("$s1, $e1, $s2, $e2, $(norm(D-H.A*H.B',2)/norm(D))")
        elseif H.m > Nleaf && H.n > Nleaf
                H.is_hmat = true
                H.children = Array{Hmat}([Hmat() Hmat()
                                        Hmat() Hmat()])
                m1 = s1 + Int(round((e1-s1)/2)) - 1
                m2 = s2 + Int(round((e2-s2)/2)) - 1
                helper(H.children[1,1], s1, m1, s2, m2)
                helper(H.children[1,2], s1, m1, m2+1, e2)
                helper(H.children[2,1], m1+1, e1, s2, m2)
                helper(H.children[2,2], m1+1, e1, m2+1, e2)
        else
            H.is_fullmatrix = true
            H.C = full_mat(f, collect(s1:e1)*h, collect(s2:e2)*h)
        end
        
    end
    H = Hmat()
    helper(H, N1, N2, N1, N2)
    return H
end

# Linear Construction Time!
function test_construct1D_low_rank()
    for n = [10,11,12,13,14]
        n = 12
    h = 1/2^n
    eps = 1
    x = collect(1:2^n)*h
    y = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(eps, 5)
    @time H = construct1D_low_rank(f, alpha, beta, h, 1,2^n, 64, 2^(n-2))
    # @time G = full_mat(f, x, y)
    # to_fmat!(H)
    # println(H.C)
    matshow(H)
    return
    println(norm(G-H.C,2)/norm(G,2))
    end
end


function full_mat2D(f,x,y)
    m = size(x,1)
    n = size(y,1)
    A = zeros(m, n)
    for i = 1:m
        for j = 1:n
            A[i,j] = f(x[i,:],y[j,:])
        end
    end
    return A
end

function fast_construct_rk_mat2D(alpha, beta, x, y)
    xbar = mean(x, dims=1)
    t0 = x .- xbar
    t = y .- xbar
    n = size(x,1)
    m = size(y,1)
    r = length(alpha)
    U = zeros(n, r)
    V = zeros(m, r)
    for i = 1:r
        U[:,i] = alpha[i].(t0[:,1], t0[:,2])
        V[:,i] = beta[i].(t[:,1],t[:,2])
    end
    return U, V
end

function admissible2_2d(x, y)
    xc = mean(x, dims=1)
    yc = mean(y, dims=1)
    dist = norm(xc-yc)
    D1 = 2*maximum( sqrt.(sum((xc .- x).^2, dims=2) ))
    D2 = 2*maximum( sqrt.(sum((yc .- y).^2, dims=2) ))
    if dist>=min(D1, D2)
        return true
    else
        return false
    end
end

# Given a 2^mx2^n grid, find the Z index 
function rearange2D(m, n)
    function helper(m, n)
        if m==0 
            return collect(1:2^n)
        elseif n==0
            return collect(1:2^m)
        end
        M = 2^m
        N = 2^n
        M2 = 2^(m-1)
        N2 = 2^(n-1)
        I0 = helper(m-1, n-1)
        I1 = zeros(Int64, 2^m, 2^n)
        R = M2*N2
        I1[1:M2, 1:N2] = 1:R
        I1[M2+1:end, 1:N2] = R .+ (1:R)
        I1[1:M2, N2+1:end] = 2R .+ (1:R)
        I1[M2+1:end, N2+1:end] = 3R .+ (1:R)
        I1[1:M2, 1:N2] = I1[1:M2, 1:N2][I0]
        I1[M2+1:end, 1:N2]= I1[M2+1:end, 1:N2][I0]
        I1[1:M2, N2+1:end]= I1[1:M2, N2+1:end][I0]
        I1[M2+1:end, N2+1:end]= I1[M2+1:end, N2+1:end][I0]
        return I1[:]
    end

    I = helper(m, n)
    invI = zeros(Int64, 2^m*2^n)
    for i = 1:2^m*2^n
        invI[I[i]] = i
    end
    return I, invI
end

# x should be well pre-arranged
function construct2D_low_rank(f, alpha, beta, x,  Nleaf, MaxBlock)
    function helper(H::Hmat, x, y)
        H.m = size(x,1)
        H.n = size(y,1)
        if H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
            H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
            @views begin
                x1 = x[1:div(H.m,2),:]
                x2 = x[div(H.m,2)+1:end,:]
                y1 = y[1:div(H.n,2),:]
                y2 = y[div(H.n,2)+1:end,:]
                helper(H.children[1,1], x1, y1)
                helper(H.children[1,2], x1, y2)
                helper(H.children[2,1], x2, y1)
                helper(H.children[2,2], x2, y2)
            end
        elseif admissible2_2d(x, y)
            H.is_rkmatrix = true
            H.A, H.B = fast_construct_rk_mat2D(alpha, beta, x, y)
            # H.is_fullmatrix = true
            # H.C = full_mat2D(f, x, y)
            # println("$s1, $e1, $s2, $e2, $(norm(D-H.A*H.B',2)/norm(D))")
        elseif H.m > Nleaf && H.n > Nleaf
                H.is_hmat = true
                H.children = Array{Hmat}([Hmat() Hmat()
                                        Hmat() Hmat()])
                @views begin
                    x1 = x[1:div(H.m,2),:]
                    x2 = x[div(H.m,2)+1:end,:]
                    y1 = y[1:div(H.n,2),:]
                    y2 = y[div(H.n,2)+1:end,:]
                    helper(H.children[1,1], x1, y1)
                    helper(H.children[1,2], x1, y2)
                    helper(H.children[2,1], x2, y1)
                    helper(H.children[2,2], x2, y2)
                end
        else
            H.is_fullmatrix = true
            H.C = full_mat2D(f, x, y)
        end
        
    end
    H = Hmat()
    helper(H, x, x)
    return H
end

function makegrid2D(L, n, I=nothing)
    x = LinRange(-L, L, 2^n)
    x = reverse(x)
    X = zeros(2^n*2^n,2)
    for j = 1:2^n
        for i = 1:2^n
            X[i+(j-1)*2^n,:] = [x[i]; x[j]]
        end
    end
    if I != nothing
        X = X[I,:]
    end
    return X
end


function test_rearange2D()
    n = 5
    L = 1.0
    x = LinRange(-L, L, 2^n)
    x = reverse(x)
    X = zeros(2^n*2^n,2)
    for j = 1:2^n
        for i = 1:2^n
            X[i+(j-1)*2^n,:] = [x[i]; x[j]]
        end
    end
    I, invI = rearange2D(n, n)
    # println(I)
    X = X[invI,:]
    # println(X[1:10,:])
    for i = 1:16
        scatter(X[(i-1)*64+1:i*64,1], X[(i-1)*64+1:i*64,2])
    end
    xlim([-L,L])
    ylim([-L,L])
    savefig("tmp.png")
    close("all")
end

function test_construct2D_low_rank()
    for n = [4,5,6,7,8,9,10]
        eps = 1
        f, alpha, beta = Merton_Kernel2D(eps, 5)
        function nf(x, y)
            h = 2/2^n
            if abs(x[1]-y[1])<h/4 && abs(x[2]-y[2])<h/4
                return -10/h^2+f(x,y)
            else
                return f(x,y)
            end
        end
        
        I, invI = rearange2D(n, n)
        X = makegrid2D(1.0, n, invI)
        
        t11 = @timed H = construct2D_low_rank(nf, alpha, beta, X, 16, 2^(2n-3))
        t12 = @timed G = full_mat2D(nf, X, X)
        HC = to_fmat(H)
        mat_err = norm(G-HC,2)/norm(G,2)


        y = rand(2^2n)
        w1 = zero(y)
        w2 = zero(y)
        t21 = @timed begin
            for i = 1:10
                w1 = H*y
            end
        end

        t22 = @timed begin
            for i = 1:10
                w2 = G*y
            end
        end

        matvec_err = norm(w1-w2)/norm(w2)

        t31 = @timed lu!(H)
        t32 = @timed F = lu!(G)

        t41 = @timed begin
            for i = 1:10
                w1 = H\y
            end
        end

        t42 = @timed begin
            for i = 1:10
                w2 = F\y
            end
        end

        solve_err = norm(w1-w2)/norm(w2)

        
        println("======= $(2^2n)x$(2^2n) ========")
        @printf("MatCon:  Hmat:%0.6f, Full:%0.6f      Error=%g\n", t11[2], t12[2], mat_err)
        @printf("MatVec:  Hmat:%0.6f, Full:%0.6f      Error=%g\n", t21[2]/10, t22[2]/10, matvec_err)
        @printf("LU    :  Hmat:%0.6f, Full:%0.6f              \n", t31[2], t32[2])
        @printf("Solve :  Hmat:%0.6f, Full:%0.6f      Error=%g\n", t41[2]/10, t42[2]/10, solve_err)
    end
end
