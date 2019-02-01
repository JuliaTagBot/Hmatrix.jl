export 
Cluster,
Hmat,
NewHmat,
FullMat

@with_kw mutable struct Cluster
    X::Array{Float64}  = Array{Float64}([])     # particle position
    P::Array{Int64,1} = Array{Int64}([])        # permutation
    left::Union{Cluster,Nothing} = nothing      # left child cluster
    right::Union{Cluster,Nothing} = nothing     # right child cluster
    m::Int64 = 0                                # |left child|
    n::Int64 = 0                                # |right child|
    N::Int64 = 0                                # number of points
    isleaf::Bool = false                        # if it is leaf
    s::Int64 = 0                                # start index after permutation
    e::Int64 = 0                                # end index after permutation
    center::Array{Float64} = Array{Float64}([])
    diam::Union{Nothing,Float64} = nothing
end

@with_kw mutable struct Hmat
    A::Array{Float64,2} = zeros(0,0)
    B::Array{Float64,2} = zeros(0,0)
    C::Array{Float64,2} = zeros(0,0)
    P::Array{Int64,1} = Array{Int64}([])
    is_rkmatrix::Bool = false
    is_fullmatrix::Bool = false
    is_hmat::Bool = false
    m::Int = 0
    n::Int = 0
    children::Array{Hmat} = Array{Hmat}([])
    s::Union{Cluster,Nothing} = nothing
    t::Union{Cluster,Nothing} = nothing # used in the construction phase and later abondoned
end

function NewHmat()
    c = construct_cluster(Hparams.Geom, Hparams.MinBlock)
    compute_geom_info(c)
    return c, __c1(c)
end

FullMat(F::Function, X::Array,Y::Array) = kernel_full(F, X, Y)


function __c1(c)
    MaxBlock, Kernel = Hparams.MaxBlock, Hparams.Kernel
    h = Hmat()
    function helper(H::Hmat, s::Cluster, t::Cluster)
        H.m = s.N
        H.n = t.N
        H.s = s
        H.t = t

        # Matrix Size
        if (H.m <= Hparams.MinBlock || H.n <= Hparams.MinBlock) || s.isleaf || t.isleaf
            H.is_fullmatrix = true
        elseif H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
        else
            if s==t
                if s.isleaf || t.isleaf
                    H.is_fullmatrix = true
                else
                    H.is_hmat = true
                end
            else
                if Hparams.ConsMethod == "bbfmm"
                    # dicide if the block is admissible
                    if (s.diam+t.diam)*0.55<=norm(s.center-t.center)
                        H.is_rkmatrix = true
                        U, V = bbfmm(Kernel, s.X, t.X, Hparams.MaxRank)
                    else
                        H.is_hmat = true
                    end
                elseif Hparams.ConsMethod == "separate"
                    # @show (s.diam+t.diam)/2, norm(s.center-t.center)
                    if (s.diam+t.diam)*0.55<=norm(s.center-t.center)
                        H.is_rkmatrix = true
                        U, V = _rk_matrix(Hparams.α, Hparams.β, s.X, t.X)
                    else
                        H.is_hmat = true
                    end
                elseif Hparams.ConsMethod == "aca2"
                    if (s.diam+t.diam)*0.55<=norm(s.center-t.center)
                        H.is_rkmatrix = true
                        U, V = aca2(Hparams.Kernel, s.X, t.X, Hparams.MaxRank)
                        if size(U,2)>Hparams.MaxRank
                            H.is_hmat = true
                        end
                    else
                        H.is_hmat = true
                    end
                else
                    A = kernel_full(Kernel, s.X, t.X)
                    U, V = compress(A)
                    k = size(U,2)
                    if k<=Hparams.MaxRank
                        H.is_rkmatrix = true
                    else
                        H.is_hmat = true
                    end
                    if Hparams.verbose
                        println("$(size(A)), $k($(Hparams.MaxRank)), $(H.is_rkmatrix) ")
                    end
                end
            end
            
        end

        if H.is_fullmatrix
            H.C = kernel_full(Kernel, s.X, t.X)
        elseif H.is_rkmatrix
            H.A = U
            H.B = V
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

function Base.:(==)(c1::Cluster, c2::Cluster)
    return c1.s==c2.s && c1.e==c2.e
end

function compute_geom_info(c::Cluster)
    function helper(c)
        c.center = sum(c.X, dims=1)/size(c.X,1)
        diam = c.X - repeat(reshape(c.center, 1, size(c.X,2)),size(c.X,1),1)
        c.diam = 2sqrt(maximum(sum(diam.^2, dims=2)))
        if c.left!=nothing
            helper(c.left)
        end
        if c.right!=nothing
            helper(c.right)
        end
    end
    helper(c)
end

function _rk_matrix(alpha, beta, x, y)
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


#====== Legacy codes ======#
function bisect_cluster(X::Array{Float64})
    clf = cluster.KMeans(n_clusters=2, random_state=0)
    clf[:fit](X)
    L = clf[:labels_]
    P1 = findall(L .== 0)
    P2 = findall(L .== 1)
    X1 = X[P1,:]
    X2 = X[P2,:]
    return X1, P1, X2, P2
end


function inverse_permutation(P::AbstractArray{Int64})
    Q = copy(P)
    for i = 1:length(P)
        Q[P[i]] = i
    end
    return Q
end

function construct_cluster(X::Array{Float64}, Nleaf::Int64)
    np.random[:seed](2333)
    if size(X,2)==1
        X = reshape(X,length(X),1)
    end

    function downward(c::Cluster)
        if c.N<=Nleaf
            c.isleaf = true
            return
        end
        X1, P1, X2, P2 = bisect_cluster(c.X)
        P1 = c.P[P1]
        P2 = c.P[P2]
        c.left = Cluster(X = X1, P = P1, N = length(P1), s = c.s, e = c.s+length(P1)-1)
        c.right = Cluster(X = X2, P = P2, N = length(P2), s = c.s+length(P1), e = c.e)
        c.m = c.left.N
        c.n = c.right.N
        downward(c.left)
        downward(c.right)
    end

    function upward(c::Cluster)
        if c.isleaf
            return
        end
        upward(c.left)
        upward(c.right)
        c.P = [c.left.P;c.right.P]
        c.X = [c.left.X;c.right.X]
    end

    c = Cluster(X = X, P = collect(1:size(X,1)), N = size(X,1), s = 1, e = size(X,1))
    downward(c)
    upward(c)
    return c
end

function kernel_full(f::Function, X::Array{Float64}, Y::Array{Float64})
    if size(X,2)==1
        A = zeros(length(X), length(Y))
        for i = 1:length(X)
            for j = 1:length(Y)
                A[i,j] = f(X[i], Y[j])
            end
        end
    else
        A = zeros(size(X,1), size(Y,1))
        for i = 1:size(X,1)
            for j = 1:size(Y,1)
                A[i,j] = f(X[i,:], Y[j,:])
            end
        end
    end
    return A
end

function kernel_svd(f::Function, X::Array{Float64}, Y::Array{Float64})
    A = kernel_full(f, X, Y)
    U,S,V = svd(A)
    k = rank_truncate(S)
    return k, U, S, V
end

function construct_hmat(f::Function, c::Cluster, Nleaf::Int64, Rrank::Int64, MaxBlock::Int64)
    if MaxBlock==-1
        MaxBlock = div(size(c.X,1), 4)
    end
    h = Hmat()
    function helper(H::Hmat, s::Cluster, t::Cluster)
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
            A = kernel_full(f, s.X, t.X)
            U, V = compress(A)
            k = size(U,2)
            if k<=Rrank
                H.is_rkmatrix = true
            else
                H.is_hmat = true
            end
            # println("$(size(A)), $Rrank, $k ")
            
        end

        if H.is_fullmatrix
            H.C = kernel_full(f, s.X, t.X)
        elseif H.is_rkmatrix
            H.A = U
            H.B = V
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

construct_hmat(f::Function, X::Array{Float64}, Nleaf::Int64, Rrank::Int64,
    eps::Float64, MaxBlock::Int64, method="svd") = construct_hmat(f, construct_cluster(X, Nleaf), Nleaf, Rrank,
                                                                eps, MaxBlock, method)
    

function construct_hmat(A::Array{Float64}, c::Cluster, Nleaf::Int64, Rrank::Int64,
                eps::Float64, MaxBlock::Int64)
    if MaxBlock == -1
        MaxBlock = Int(round(size(A,1)/4))
    end
    P = c.P
    if length(c.P)>0
        A = A[P,P]
    end

    h = Hmat()
    function helper(H::Hmat, s::Cluster, t::Cluster)
        H.m = s.N
        H.n = t.N
        H.s = s
        H.t = t

        # println(s.e-s.s+1, t.e-t.s+1)

        # Matrix Size
        if (H.m <= Nleaf || H.n <= Nleaf) || s.isleaf || t.isleaf
            H.is_fullmatrix = true
        elseif H.m > MaxBlock || H.n > MaxBlock
            H.is_hmat = true
        else
            # Rank consideration
            M = A[s.s:s.e, t.s:t.e]
            U,S,V = svd(M)
            k = rank_truncate(S, eps)

            # println("* $(size(M)), $k, $Rrank")
            if k<=Rrank
                H.is_rkmatrix = true
            else
                H.is_hmat = true
            end
        end

        if H.is_fullmatrix
            H.C = A[s.s:s.e, t.s:t.e]
        elseif H.is_rkmatrix
            # @assert k!=0
            H.A = U[:,1:k]
            H.B = V[:,1:k] * diagm(0=>S[1:k])
            
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
    return h, P
end
function construct_hmat(A::Array{Float64}, X::Array{Float64}, Nleaf::Int64, Rrank::Int64,
                eps::Float64, MaxBlock::Int64)
    c = construct_cluster(X, Nleaf)
    return construct_hmat(A, c, Nleaf, Rrank, eps, MaxBlock)
end


function cluster_from_list(l::Array{Int64}, X::Union{Nothing, Array{Float64}}=nothing)
    function helper(cs)
        if length(cs)==1
            return cs[1]
        end
        next_list = []
        
        for i = 1:2:length(cs)-1
            c1 = cs[i]
            c2 = cs[i+1]
            c = Cluster(m = c1.N, n = c2.N, left = c1, right = c2, N = c1.N + c2.N, isleaf = false, s = c1.s, e = c2.e, P=c1.s:c2.e)
            if X!=nothing
                c.X = X[c.s:c.e,:]
            end
            push!(next_list, c)
        end
        if mod(length(cs),2)==1
            push!(next_list, cs[end])
        end
        return helper(next_list)
    end
    cs = []
    s = 1
    for r in l
        c = Cluster(N = r, isleaf = true, s = s, e = s+r-1, P=s:s+r-1)
        if X!=nothing
            c.X = X[c.s:c.e,:]
        end
        push!(cs, c)
        s = s+r
    end
    return helper(cs)
end

function uniform_cluster(l, N)
    n = div(l, N)
    C = ones(Int, n)*N
    if sum(C)<l
        C = [C;l-sum(C)]
    end
    return C
end


# ============= testing ================

function easy_construct_cluster(A::Array{Float64})
    n = size(A,1)
    C = uniform_cluster(n, 64)
    return C
end

function test_matvec()
    f = (x,y)->1/(norm(x-y)+0.1)
    X = rand(2000,2);
    MaxBlock = div(size(X,1),4)
    println("*** Test begin")
    for eps = [ 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        println("eps = $eps")
        H,P=construct_hmat(f, X, 64, 5, eps, MaxBlock);
        Y = X[P,:];
        A = kernel_full(f, Y, Y)
        println("Matrix Error = ", pointwise_error(A, to_fmat(H)))
        x = rand(size(X,1))
        y1 = A*x
        y2 = H*x
        println("Matrix Vector Error = ", rel_error(y2, y1))

        # matshow(H)
        # savefig("H.png")
        # close("all")
        if eps<=1e-2
            HH = to_fmat(H)
            lu!(H, eps)
            y1 = H\x
            y2 = A\x
            println("Solve Error = ", rel_error(y2, y1))
            C = to_fmat(H)
            G = A[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
            println("LU Matrix Error = $(maximum(abs.(G)))")

            G = HH[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
            println("LU Operator Error = $(maximum(abs.(G)))")
        end
    end
    println("====================")
    for N = [1, 3, 5, 10, 20, 30]
        println("N = $N")
        H,P=construct_hmat(f, X, 64, N, 1e-3, MaxBlock);
        Y = X[P,:];
        A = kernel_full(f, Y, Y)
        println("Matrix Error = ", pointwise_error(A, to_fmat(H)))
        x = rand(size(X,1))
        y1 = A*x
        y2 = H*x
        println("Matrix Vector Error = ", rel_error(y2, y1))

        HH = to_fmat(H)
        lu!(H, 1e-3)
        y1 = H\x
        y2 = A\x
        println("Solve Error = ", rel_error(y2, y1))
        C = to_fmat(H)
        G = A[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
        println("LU Matrix Error = $(maximum(abs.(G)))")

        G = HH[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
        println("LU Operator Error = $(maximum(abs.(G)))")
    end


end

function test_single_case()
    f = (x,y)->1/(norm(x-y)+0.1)
    X = rand(10000,3);
    MaxBlock = div(size(X,1),4)
    println("*** Test begin")
    eps = 1e-4
    H,P=construct_hmat(f, X, 64, 5, eps, MaxBlock);
    Y = X[P,:];
    A = kernel_full(f, Y, Y)
    println("Matrix Error = ", pointwise_error(A, to_fmat(H)))
    x = rand(size(X,1))
    y1 = A*x
    y2 = H*x
    println("Matrix Vector Error = ", rel_error(y2, y1))

    matshow(H)
    savefig("H.png")
    close("all")
    lu!(H)
    y1 = H\x
    y2 = A\x
    println("Solve Error = ", rel_error(y2, y1))
    C = to_fmat(H)
    G = A[H.P,:] - (LowerTriangular(C)-diagm(0=>diag(C))+UniformScaling(1.0))*UpperTriangular(C)
    println("LU Matrix Error = $(maximum(abs.(G)))")
end