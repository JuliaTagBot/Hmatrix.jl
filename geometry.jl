# geometry.jl
# This file contains the utilites for Cluster struct. It also contains basic functions to generate H-matrix
using PyCall
using LinearAlgebra
@pyimport sklearn.cluster as cluster

function aca(A::Array{Float64}, eps::Float64, Rrank::Int64)
    U = zeros(size(A,1), Rrank)
    V = zeros(size(A,2), Rrank)
    R = ccall((:aca_wrapper,"/Users/kailaix/Desktop/hmat/third-party/build/libaca.dylib"), Cint, (Ref{Cdouble}, Ref{Cdouble},
                Ref{Cdouble},Cint, Cint,Cdouble, Cint ), A, U, V, size(A,1), size(A,2), eps, Rrank)
    U = U[:,1:Rrank]
    V = V[:,1:Rrank]
    return U, V
end

# column pivoting RRQR
function rrqr(A,tol)
    m, n = size(A)
    Q = zeros(m,n)
    R = zeros(m,n)
    rank = 0
    ind = collect(1:n)
    Acopy = copy(A)
    
    nrm = norm(A,2)
    for r in 1:min(m,n)
        cn = zeros(n,1)
        for j=r:n
            cn[j] = norm(A[:,j])
        end
        tau = maximum(cn)
        if tau!=0
            k = argmax(cn[:])
        end
        if k>r
            tmp = ind[r]; ind[r] = ind[k]; ind[k] = tmp;
            tmp = A[:,r]; A[:,r] = A[:,k]; A[:,k] = tmp;
            tmp = R[1:r-1,r]; R[1:r-1,r] = R[1:r-1,k]; R[1:r-1,k] = tmp;
        end
        R[r,r] = tau
        
        #if (R[1,1]<=tol || R[r,r]/R[1,1]<=tol)
        if (R[1,1]<=tol || norm(triu(A[r+1:end,r+1:end]))<=tol)
        #if (R[1,1]<=tol || vecnorm(triu(A[r+1:end,r+1:end]))/nrm<=tol) # Golub van Loan section 5.5.7 eq 5.5.6
            r -= 1
            rank = r
            break;
        end
        Q[:,r] = A[:,r]/R[r,r];
        R[r,r+1:n] = Q[:,r]' * A[:,r+1:n]
        A[:,r+1:n] = A[:,r+1:n] - Q[:,r]*R[r,r+1:n]'
        rank = r
    end
    
    if rank==0
        Q = []
        R = []
    else
        Q = Q[:,1:rank]
        R = R[1:rank,:]
    end
    iind = inverse_permutation(ind)
    return Q, R[:,iind]'
end


function rank_truncate(S, eps=1e-10)
    if length(S)==0
        return 0
    end
    k = findlast(S/S[1] .> eps)
    if isa(k, Nothing)
        return 0
    else
        return k
    end
end

# C is a full matrix
# the function will try to compress C with given tolerance eps
# Rrank is required when method = "aca"
function compress(C, eps=1e-10, method="svd"; Rrank = nothing)
    # if the matrix is a zero matrix, return zero vectors
    if sum(abs.(C))≈0
        A = zeros(size(C,1),1)
        B = zeros(size(C,2),1)
        return A, B
    end

    if method=="svd"
        if size(C,1)==size(C,2)
            U,S,V = psvd(C)    # fast svd is availabel
        else
            U,S,V = svd(C)
        end
        k = findlast(S/S[1] .> eps)
        @assert !isa(k, Nothing) # k should never be zero
        A = U[:,1:k]
        B = (diagm(0=>S[1:k])*V'[1:k,:])'
        return A, B
    end

    if method=="aca"
        @assert !isa(Rrank, Nothing)
        U,V = aca(C, eps,Rrank);
        return U,V
    end

    if method=="rrqr"
        U,V = rrqr(C, eps)
        return U,V
    end

    error("Method $method not implemented yet")

end


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
            A = kernel_full(f, s.X, t.X)
            U, V = compress(A, eps, method, Rrank=2*Rrank)
            k = size(U,2)
            if k<=Rrank
                H.is_rkmatrix = true
            else
                H.is_hmat = true
            end
            
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


# ============= testing ================
function test_matvec()
    f = (x,y)->1/(norm(x-y)+0.1)
    X = rand(2000,2);
    MaxBlock = div(size(X,1),4)
    println("*** Test begin")
    for eps = [1e-1, 1e-3, 1e-6, 1e-8]
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
            lu!(H)
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
        lu!(H)
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