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

function admissible2(s1, e1, s2, e2)
    p = e1-s1
    if s1>=e2 || s2>=e1
        if s1>e2
            d = s1-e2
        else
            d = s2-e1
        end
        # return d>p
        return true
    else
        return false
    end


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
    h = 1/2^n
    eps = 1
    x = collect(1:2^n)*h
    y = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(eps, 10)
    @time H = construct1D_low_rank(f, alpha, beta, h, 1,2^n, 64, 2^(n-3))
    @time G = full_mat(f, x, y)
    to_fmat!(H)
    # println(H.C)
    # matshow(H)
    println(norm(G-H.C,2)/norm(G,2))
    end
end