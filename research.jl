
#  research script for Merton kernel
include("hmat.jl")


function fast_construct_rk_mat(alpha, beta, x, y)
    xbar = mean(x, dims=1)
    t0 = x .- xbar
    t = y .- xbar
    n = size(x,1)
    m = size(y,1)
    r = length(alpha)
    U = zeros(n, r)
    V = zeros(m, r)
    for i = 1:r
        if size(x,2)==1
            U[:,i] = alpha[i].(t0)
            V[:,i] = beta[i].(t)
        elseif size(x,2)==2
            U[:,i] = alpha[i].(t0[:,1], t0[:,2])
            V[:,i] = beta[i].(t[:,1],t[:,2])
        else
            error("Not implemented yet")
        end
    end
    return U, V
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

function Merton_Kernel2D(e, r)
    f = (x,y)->exp(-e^2*norm(x-y)^2)
    alpha = []
    beta = []
    for m = 0:r-1
        for n = 0:r-1
            push!(alpha, (s,t)->mk2d_1(t, s, m, n, e))
            push!(beta, (s,t)->mk2d_2(t, s, m, n, e))
        end
    end
    return f, alpha, beta
end

function strong_admissible(X::Array{Float64}, Y::Array{Float64})
    x = mean(X,dims=1)
    y = mean(Y,dims=1)
    dist = norm(x-y)
    d1 = maximum(maximum(X, dims=1)-minimum(X, dims=1))
    d2 = maximum(maximum(Y, dims=1)-minimum(Y, dims=1))
    dist = dist - (d1+d2)/2.0
    # println(dist, " ", d1 , " ", d2)
    if dist >= max(d1, d2)
        return true
    else
        return false
    end
end

function uniform_cluster_1D(n, h, Nleaf)
    C = uniform_cluster(n, Nleaf)
    c = cluster_from_list(C)
    function helper(c)
        c.X = h*collect(c.s:c.e)
        c.X = reshape(c.X, length(c.X), 1)
        if c.left!=nothing
            helper(c.left)
            helper(c.right)
        end
    end
    helper(c)
    return c
end

function construct_hmat_from_expansion(f::Function, αs, βs, c::Cluster, Nleaf::Int64, MaxBlock::Int64)
    h = Hmat()
    function helper(H::Hmat, s::Cluster, t::Cluster)
        # @show s.s, s.e, t.s, t.e
        
        H.m = s.N
        H.n = t.N
        H.s = s
        H.t = t

        # Matrix Size
        if (H.m <= Nleaf || H.n <= Nleaf) || s.isleaf || t.isleaf
            H.is_fullmatrix = true
        elseif H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
        else
            if strong_admissible(s.X, t.X)
                H.is_rkmatrix = true
            else
                H.is_hmat = true
            end
        end

        if H.is_fullmatrix
            H.C = kernel_full(f, s.X, t.X)
        elseif H.is_rkmatrix
            H.A, H.B = fast_construct_rk_mat(αs, βs, s.X, t.X)
            # println("Low Rank Matrix: $(H.m)x$(H.n)")
        else
            H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
            helper(H.children[1,1], s.left, t.left)
            helper(H.children[1,2], s.left, t.right)
            helper(H.children[2,1], s.right, t.left)
            helper(H.children[2,2], s.right, t.right)
        end

    end
    helper(h, c, c)
    return h
end

# ================ test ===================
function test_case_1()
    # for n = [10,11,12,13,14]
    n = 12
    h = 1/2^n
    x = collect(0:2^n-1)*h
    c = construct_cluster(x, 64)
    # c = uniform_cluster_1D(2^n, h, 64)
    f, αs, βs = Merton_Kernel(1.0, 5)
    H = construct_hmat_from_expansion(f, αs, βs, c, 64, 2^(n-2))
    # @time G = full_mat(f, x, y)
    # to_fmat!(H)
    # println(H.C)
    matshow(H)
    C = to_fmat(H)
    G = kernel_full(f, c.X, c.X)
    println(pointwise_error(C, G))
    # end
end

function test_case_2(n = 10, tol=1e-5; rundense=false)
    # for n = [10,11,12,13,14]
    
    h = 1/2^n
    X = collect(0:2^n-1)*h
    c = construct_cluster(X, 64)
    # c = uniform_cluster_1D(2^n, h, 64)
    f, αs, βs = Merton_Kernel(1.0, 5)
    function new_f(x,y)
        if abs(y-x)<h
            return 10
        end
        return f(x,y)
    end
    H = construct_hmat_from_expansion(new_f, αs, βs, c, 64, 2^(n-2))
    # @time G = full_mat(f, x, y)
    # to_fmat!(H)
    # println(H.C)
    # matshow(H)
    A = kernel_full(new_f, c.X, c.X)
    # println(pointwise_error(C, G))
    # end

    println("eps = $tol")
    println("Matrix Error = ", pointwise_error(A, to_fmat(H)))
    x = rand(size(X,1))
    y1 = A*x
    y2 = H*x
    println("Matrix Vector Error = ", rel_error(y2, y1))

    if rundense
        B = copy(A)
        @time lu!(B)
    end
    
    HH = to_fmat(H)
    @time lu!(H, tol)
    y1 = H\x
    y2 = A\x
    println("Solve Error = ", rel_error(y2, y1))
    C = to_fmat(H)
    G = A[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
    println("LU Matrix Error = $(maximum(abs.(G)))")

    G = HH[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
    println("LU Operator Error = $(maximum(abs.(G)))")

    
end