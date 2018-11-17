# geometry.jl
# This file contains the utilites for Cluster struct. It also contains basic functions to generate H-matrix
@pyimport sklearn.cluster as cluster

function bisect_cluster(X::Array{Float64})
    # code here
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
    function downward(c::Cluster)
        if c.N<=Nleaf
            c.isleaf = true
            return
        end
        X1, P1, X2, P2 = bisect_cluster(c.X)
        P1 = c.P[P1]
        P2 = c.P[P2]
        c.left = Cluster(X = X1, P = P1, N = length(P1), s = c.s, e = c.s+length(P1)-1)
        c.right = Cluster(X = X2, P = P2, N = length(P2), s = c.s+length(P1)-1, e = c.e)
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
    A = zeros(size(X,1), size(Y,1))
    for i = 1:size(X,1)
        for j = 1:size(Y,1)
            A[i,j] = f(X[i,:], Y[j,:])
        end
    end
    return A
end

function kernel_svd(f::Function, X::Array{Float64}, Y::Array{Float64}, eps::Float64)
    A = kernel_full(f, X, Y)
    U,S,V = svd(A)
    k = rank_truncate(S, eps)
    return k, U, S, V
end

#TODO:
function kernel_bbfmm(f::Function, X::Array{Float64}, Y::Array{Float64}, Rrank::Int64, eps::Float64)
    x, _ = gausschebyshev(Rrank)
    # ...
end

#TODO:
function kernel_mvd(f::Function, X::Array{Float64}, Y::Array{Float64}, Rrank::Int64, eps::Float64)
end


function construct_hmat(f::Function, X::Array{Float64}, Nleaf::Int64, Rrank::Int64,
    eps::Float64, MaxBlock::Int64, method="svd")
    c = construct_cluster(X, Nleaf)
    P = c.P
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
            if method=="svd"
                k, U, S, V = kernel_svd(f, s.X, t.X, eps)
                if k<=Rrank
                    H.is_rkmatrix = true
                else
                    H.is_hmat = true
                end
            elseif method=="bbfmm"
                err, U, V = kernel_bbfmm(f, X, Y, Rrank, eps)
                # #TODO:if err ...
            elseif method=="expansion"
                err, U, V = kernel_expansion(f, X, Y, Rrank, eps)
                # #TODO:if err ...
            end
        end

        if H.is_fullmatrix
            H.C = kernel_full(f, s.X, t.X)
        elseif H.is_rkmatrix
            if method=="svd"
                # @assert k!=0
                H.A = U[:,1:k]
                H.B = V[:,1:k] * diagm(0=>S[1:k])
                println("$(size(H)), $k, $Rrank => LR")
            elseif method=="bbfmm"
                H.A = U
                H.B = V
            elseif method=="expansion"
                #TODO:
            end
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

            println("* $(size(M)), $k, $Rrank")
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


function cluster_from_list(l)
    function helper(cs)
        if length(cs)==1
            return cs[1]
        end
        next_list = []
        
        for i = 1:2:length(cs)-1
            c1 = cs[i]
            c2 = cs[i+1]
            c = Cluster(m = c1.N, n = c2.N, left = c1, right = c2, N = c1.N + c2.N, isleaf = false, s = c1.s, e = c2.e)
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
        push!(cs, Cluster(N = r, isleaf = true, s = s, e = s+r-1))
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