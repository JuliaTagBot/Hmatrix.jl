# hmat.jl
# This file contains the utilites for H-matrix


using Parameters
using LinearAlgebra
using PyPlot
using Printf
using Statistics
using Profile
using LowRankApprox
using FastGaussQuadrature
using TimerOutputs
using PyCall

if !@isdefined Cluster
    @with_kw mutable struct Cluster
        X::Array{Float64}  = Array{Float64}([])
        P::Array{Int64,1} = Array{Int64}([])
        left::Union{Cluster,Nothing} = nothing
        right::Union{Cluster,Nothing} = nothing
        m::Int64 = 0
        n::Int64 = 0
        N::Int64 = 0
        isleaf::Bool = false
        s::Int64 = 0
        e::Int64 = 0 # start and end index after permutation
    end
end

if !@isdefined Hmat
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
end

include("geometry.jl")
include("harithm.jl")

function Base.:print(c::Cluster)
    current_level = [c]
    while length(current_level)>0
        println(join(["$(x.N)($(x.s),$(x.e))" for x in current_level], " "))
        next_level = []
        for n in current_level
            if !n.isleaf 
                push!(next_level, n.left)
                push!(next_level, n.right)
            end
            current_level = next_level
        end
    end
end

function Base.:print(h::Hmat)
    G = to_fmat(h)
    printmat(G)
end

function PyPlot.:plot(c::Cluster)
    
    function helper(level)
        clevel = 0
        current_level = [c]
        while clevel!=level && length(current_level)>0
            next_level = []
            for n in current_level
                if !n.isleaf 
                    push!(next_level, n.left)
                    push!(next_level, n.right)
                end
                current_level = next_level
            end
            clevel += 1
        end
        if length(current_level)==0
            return false
        end
        figure()
        for c in current_level
            if size(c.X,2)==1
                scatter(c.X[:,1], ones(size(c.X,1)), marker = ".")
            elseif size(c.X, 2)==2
                scatter(c.X[:,1], c.X[:,2], marker = ".")
            elseif size(c.X,2)==3
                scatter3D(c.X[:,1], c.X[:,2], c.X[:,3], marker = ".")
            end
        end
        title("Level = $level")
        savefig("level$level.png")
        close("all")
        return true
    end
    flag = true
    l = 0
    while flag
        flag = helper(l)
        l += 1
    end
end

# const tos = TimerOutput()

function Base.:size(H::Hmat)
    return (H.m, H.n)
end

function Base.:size(H::Hmat, i::Int64)
    if i==1
        return H.m
    elseif i==2
        return H.n
    else
        @error "size(Hmat, Int64): invalid dimension $i"
    end
end


# utilities
function consistency(H)
    if H.is_rkmatrix + H.is_fullmatrix + H.is_hmat != 1
        @error "Matrix is ambiguous or unset"
    end

    if H.m == 0 || H.n == 0 
        @error "Empty matrix"
    end

    if H.s.N!=H.m || H.t.N !=H.n
        @error "Matrix dimensions are not consistent with cluster"
    end

    if H.is_rkmatrix
        if length(H.C)>0
            @error "Rank matrix should not have nonempty C"
        end
        if size(H.A, 1)!= H.m || size(H.B, 1)!=H.n || size(H.A,2)!=size(H.B,2)
            @error "Rank matrix dimension not match"
        end
        if length(H.A)==0 || length(H.B)==0
            @error "Empty rank matrices"
        end
    end
    
    if H.is_fullmatrix
        if length(H.A)>0 || length(H.B)>0
            @error "Full matrix should not have nonempty A, B"
        end
        if size(H.C,1)!=H.m || size(H.C,2)!=H.n
            @error "Full matrix dimension not match"
        end
    end

    if H.is_hmat
        if length(H.A)>0 || length(H.B)>0 || length(H.C)>0
            @error "Hmatrix should not have nonempty A, B, C"
        end
        if length(H.children)!=4
            @error "Hmatrix no children"
        end
        for i = 1:2
            for j =1:2
                consistency(H.children[i,j])
            end
        end
    end
end


function info(H::Hmat)
    dmat::Int64 = 0
    rkmat::Int64 = 0
    level::Int64 = 0
    compress_ratio::Float64 = 0
    function helper(H::Hmat, l::Int)
        # global dmat
        # global rkmat
        # global level
        if H.is_fullmatrix
            dmat += 1
            level = max(l, level)
            compress_ratio += H.m*H.n
        elseif H.is_rkmatrix
            rkmat += 1
            level = max(l, level)
            compress_ratio += size(H.A,1)*size(H.A,2) + size(H.B,1)*size(H.B, 2)
        else
            for i = 1:2
                for j = 1:2
                    helper(H.children[i,j], l+1)
                end
            end
        end
    end
    helper(H, 1)
    return dmat, rkmat, level, compress_ratio/H.m/H.n
end

#=
function fmat(A::Array{Float64})
    H = Hmat(m = size(A,1), n = size(A,2))
    H.is_fullmatrix = true
    H.C = copy(A)
    return H
end
=#

function rkmat(A, B, s, t)
    return Hmat(m = size(A,1), n = size(B,1), s=s, t=t, is_rkmatrix = true, A = A, B = B)
end

function fullmat(A, s, t)
    return Hmat(m = size(A,1), n = size(A,2), s = s, t = t, is_fullmatrix = true, C = A)
end

function hmat_from_children(h11, h12, h21, h22, s, t)
    H = Hmat(m = h11.m + h21.m, n = h11.n + h12.n, s=s, t = t, is_hmat = true)
    H.children = [h11 h12
                h21 h22]
    return H
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

function compress(C, eps=1e-10, N = nothing)
    if sum(abs.(C))≈0
        A = zeros(size(C,1),1)
        B = zeros(size(C,2),1)
        return A, B
    end

    if size(C,1)==size(C,2)
        U,S,V = psvd(C) 
    else
        U,S,V = svd(C)
    end
    if N==nothing
        N = length(S)
    end
    k = rank_truncate(S,eps)
    if k>N
        k = length(S)
    end
    A = U[:,1:k]
    B = (diagm(0=>S[1:k])*V'[1:k,:])'
    return A, B
end

function svdtrunc(A1, B1, A2, B2, eps=1e-10)
    if size(A2,2)==0 
        @assert size(B2,2)==0
        return A1, B1
    end

    if size(A1,2)==0
        @assert size(B1,2)==0
        return A2, B2
    end
    
    FA = qr([A1 A2])
    FB = qr([B1 B2])
    U,S,V = svd(FA.R*FB.R')
    k = rank_truncate(S, eps)
    A = FA.Q * U[:,1:k] * diagm(0=>S[1:k])
    B = FB.Q * V[:,1:k]
    return A, B
end

# TODO: more rank matrix addition algorithms
function rkmat_add!(a, b, scalar, method=1, eps=1e-10)
    if method==1
        a.A, a.B = svdtrunc(a.A, a.B, scalar*b.A, b.B, eps)
    else
        error("Method not defined!")
    end
end

function Base.:*(a::Hmat, v::AbstractArray{Float64})
    r = zeros(a.m, size(v,2))
    for i = 1:size(v,2)
        @views hmat_matvec!(r[:,i], a, v[:,i], 1.0)
    end
    return r
end

function matvec(a::Hmat, v::AbstractArray{Float64}, P::Array{Int64}, Q::Array{Int64})
    v = v[P,:]
    r = a*v
    r = r[Q,:]
    return r
end

# r = r + s*a*v
function hmat_matvec!(r::AbstractArray{Float64}, a::Hmat, v::AbstractArray{Float64}, s::Float64)
    if a.is_fullmatrix
        BLAS.gemm!('N','N',s,a.C,v,1.0,r)
    elseif a.is_rkmatrix
        BLAS.gemm!('N','N',s,a.A, a.B'*v,1.0,r)
    else
        m, n = a.children[1,1].m, a.children[1,1].n
        @views begin
            hmat_matvec!(r[1:m], a.children[1,1], v[1:n], s)
            hmat_matvec!(r[1:m], a.children[1,2], v[n+1:end], s)
            hmat_matvec!(r[m+1:end], a.children[2,1], v[1:n], s)
            hmat_matvec!(r[m+1:end], a.children[2,2], v[n+1:end], s)
        end
    end
end

# copy the hmatrix A to H in place.
function hmat_copy!(H::Hmat, A::Hmat)
    H.m = A.m
    H.n = A.n
    H.s = A.s
    H.t = A.t
    H.P = copy(A.P)
    if A.is_fullmatrix
        H.C = copy(A.C)
        H.is_fullmatrix = true
    elseif A.is_rkmatrix
        H.A = copy(A.A)
        H.B = copy(A.B)
        H.is_rkmatrix = true
    else
        H.is_hmat = true
        H.children = Array{Hmat}([Hmat() Hmat()
                                  Hmat() Hmat()])
        for i = 1:2
            for j = 1:2
                hmat_copy!(H.children[i,j], A.children[i,j])
            end
        end
    end
end

function Base.:copy(H::Hmat)
    G = Hmat()
    hmat_copy!(G, H)
    return G
end

# convert matrix A to full matrix
function to_fmat!(A::Hmat)
    if A.is_fullmatrix
        return
    elseif A.is_rkmatrix
        A.C = A.A*A.B'
        A.A, A.B = zeros(0,0), zeros(0,0)
    elseif A.is_hmat
        for i = 1:2
            for j = 1:2
                to_fmat!(A.children[i,j])
            end
        end
        A.C = [A.children[1,1].C A.children[1,2].C
                A.children[2,1].C  A.children[2,2].C]
        if length(A.children[1,1].P)>0
            A.P = [A.children[1,1].P;
                    A.children[2,2].P .+ A.children[1,1].m]
        end
        A.children = Array{Hmat}([])
    end
    A.is_fullmatrix = true
    A.is_rkmatrix = false
    A.is_hmat = false
end

function to_fmat(A::Hmat)
    B = copy( A)
    to_fmat!(B)
    return B.C
end

function getl(A, unitdiag)
    if unitdiag
        return LowerTriangular(A)+LowerTriangular(-diagm(0=>diag(A)) + UniformScaling(1.0))
    else
        return LowerTriangular(A)
    end
end

function getu(A, unitdiag)
    if unitdiag
        return UpperTriangular(A)+UpperTriangular(-diagm(0=>diag(A)) + UniformScaling(1.0))
    else
        return UpperTriangular(A)
    end
end

function transpose!(a::Hmat)
    a.m, a.n = a.n, a.m
    a.s, a.t = a.t, a.s
    if a.is_rkmatrix
        a.A, a.B = a.B, a.A
    elseif a.is_fullmatrix
        a.C = a.C'
    else
        for i = 1:2
            for j = 1:2
                transpose!(a.children[i,j])
            end
        end
        a.children[1,2], a.children[2,1] = a.children[2,1], a.children[1,2]
    end
end



# # special function for computing c = c - a*b
function hmat_sub_mul!(c::Hmat, a::Hmat, b::Hmat)
    hmat_add!(c, a*b, -1.0)
end

# solve a x = b where a is possibly a H-matrix. a is lower triangular. 
function mat_full_solve(a::Hmat, b::AbstractArray{Float64}, unitdiag)
    if a.is_rkmatrix
        error("A should not be a low-rank matrix")
    end
    if unitdiag
        cc = 'U'
    else
        cc = 'N'
    end
    if a.is_fullmatrix 
        LAPACK.trtrs!('L', 'N', cc, a.C, b)
    else
        # x = getl(to_fmat(a), unitdiag)\b
        # b[:] = x
        # return
        a11 = a.children[1,1]
        a21 = a.children[2,1]
        a22 = a.children[2,2]
        n = a11.m
        b1 = b[1:n,:]
        b2 = b[n+1:end, :]
        mat_full_solve(a11, b1, unitdiag)
        b2 -= a21 * b1
        mat_full_solve(a22, b2, unitdiag)
        b[:] = [b1;b2]
        # println("Error = ", maximum(abs.(b-x)))
    end
end

# Solve AX = B and store the result into B
# A, B have been prepermuted and therefore this function should not worry about permutation
function hmat_trisolve!(a::Hmat, b::Hmat, islower, unitdiag)
    # the coefficient matrix cannot be a low rank matrix,
    if a.is_rkmatrix
        error("A should not be a low-rank matrix")
    end

    # unit diagonal part or not, for blas routines
    if unitdiag
        cc = 'U'
    else
        cc = 'N'
    end

    if islower
        if a.is_fullmatrix && b.is_fullmatrix
            LAPACK.trtrs!('L', 'N', cc, a.C, b.C)
        elseif a.is_fullmatrix && b.is_rkmatrix
            if size(b.A,1)==0
                @warn "b is an empty matrix"
                return
            end
            LAPACK.trtrs!('L', 'N', cc, a.C, b.A)
        elseif a.is_fullmatrix && b.is_hmat
            error("This is never used")
        elseif a.is_hmat && b.is_hmat
            # println("HH")
            # p = getl(to_fmat(a), unitdiag)\to_fmat(b)
            a11, a12, a21, a22 = a.children[1,1], a.children[1,2],a.children[2,1],a.children[2,2]
            b11, b12, b21, b22 = b.children[1,1], b.children[1,2],b.children[2,1],b.children[2,2]
            
            # p = getl(to_fmat(a11), unitdiag)\to_fmat(b11)
            hmat_trisolve!(a11, b11, islower, unitdiag)
            # println("*** I Error = ", pointwise_error(p, to_fmat(b11)))

            # p = getl(to_fmat(a11), unitdiag)\to_fmat(b12)
            hmat_trisolve!(a11, b12, islower, unitdiag)
            # println("*** II Error = ", pointwise_error(p, to_fmat(b12)))

            # p = to_fmat(b21)-to_fmat(a21)*to_fmat(b11)
            hmat_sub_mul!(b21, a21, b11)
            # println("*** III Error = ", pointwise_error(p, to_fmat(b21)))

            # p = to_fmat(b22)-to_fmat(a21)*to_fmat(b12)
            hmat_sub_mul!(b22, a21, b12)
            # println("*** IV Error = ", pointwise_error(p, to_fmat(b22)))

            # p = getl(to_fmat(a22), unitdiag)\to_fmat(b21)
            hmat_trisolve!(a22, b21, islower, unitdiag)
            # println("*** V Error = ", pointwise_error(p, to_fmat(b21)))

            # p = getl(to_fmat(a22), unitdiag)\to_fmat(b22)
            hmat_trisolve!(a22, b22, islower, unitdiag)
            # println("*** VI Error = ", pointwise_error(p, to_fmat(b22)))

            # println("*** H Error = ", pointwise_error(p, to_fmat(b)))
        elseif a.is_hmat && b.is_fullmatrix
            # println("HF")
            # p = getl(to_fmat(a), unitdiag)\b.C
            mat_full_solve(a, b.C, unitdiag)
            # println("*** HF Error = ", pointwise_error(p, to_fmat(b)))
        elseif a.is_hmat && b.is_rkmatrix
            # println("FH")
            # p = getl(to_fmat(a), unitdiag)\to_fmat(b)
            mat_full_solve(a, b.A, unitdiag)
            # println("*** FH Error = ", pointwise_error(p, to_fmat(b)))
            # error("Not used")
        end
    else
        transpose!(a)
        transpose!(b)
        hmat_trisolve!(a, b, true, unitdiag)
        transpose!(a)
        transpose!(b)
    end
end

# Note: we do not need to permute the cluster. The only information we will use is the 
# total number of nodes instead of relative order of the points
function permute_hmat!(H::Hmat, P::AbstractArray{Int64})
    if H.is_fullmatrix
        H.C = H.C[P,:]
    elseif H.is_rkmatrix
        H.A = H.A[P,:]
    else
        m = H.children[1,1].m
        P1 = P[1:m]
        P2 = P[m+1:end] .- m
        permute_hmat!(H.children[1,1], P1)
        permute_hmat!(H.children[1,2], P1)
        permute_hmat!(H.children[2,1], P2)
        permute_hmat!(H.children[2,2], P2)
    end
end

function LinearAlgebra.:lu!(H::Hmat)
    if H.is_rkmatrix
        error("H should not be a low-rank matrix")
    end

    if H.is_fullmatrix
        F = lu!(H.C, Val{true}())
        H.P = F.p
    else
        # G = to_fmat(H)
        # lu!(G, Val{false}())

        lu!(H.children[1,1])
        
        # lu!(G[1:H.children[1,1].m, 1:H.children[1,1].n])
        # GG = to_fmat(H)
        # println("+++", maximum(abs.(G-GG)))

        permute_hmat!(H.children[1,2], H.children[1,1].P)
        # print(H)

        # E = getl(to_fmat(H.children[1,1]), true)\to_fmat(H.children[1,2])
        hmat_trisolve!(H.children[1,1], H.children[1,2], true, true)      # islower, unitdiag
        # println("Err-L ", maximum(abs.(E-to_fmat(H.children[1,2]))))
        # @assert maximum(abs.(E-to_fmat(H.children[1,2])))<1e-6


        # E = to_fmat(H.children[2,1])/getu(to_fmat(H.children[1,1]), false)
        hmat_trisolve!(H.children[1,1], H.children[2,1], false, false)   # islower, unitdiag
        # println("Err-U ", maximum(abs.(E-to_fmat(H.children[2,1]))))
        # @assert maximum(abs.(E-to_fmat(H.children[2,1])))<1e-6

        # E = to_fmat(H.children[2,2]) - to_fmat(H.children[2,1])*to_fmat(H.children[1,2])
        hmat_sub_mul!(H.children[2,2], H.children[2,1], H.children[1,2])
        # println("Err-E ", maximum(abs.(E-to_fmat(H.children[2,2]))))
        # @assert maximum(abs.(E-to_fmat(H.children[2,2])))<1e-6

        lu!(H.children[2,2])
        # print(H)

        permute_hmat!(H.children[2,1], H.children[2,2].P)
        H.P = [H.children[1,1].P; H.children[2,2].P .+ H.children[1,1].m]
        # GG = to_fmat(H)
        # println("***", size(G), maximum(abs.(G-GG)))
    end
end

# a is factorized hmatrix
function hmat_solve!(a::Hmat, y::AbstractArray{Float64}, lower=true)
    if a.is_rkmatrix
        error("a cannot be a low-rank matrix")
    end
    if lower
        if a.is_fullmatrix
            LAPACK.trtrs!('L', 'N', 'U', a.C, y)
        elseif a.is_hmat
            @views begin
                hmat_solve!(a.children[1,1], y[1:a.children[1,1].m], true)
                hmat_matvec!(y[a.children[1,1].m+1:end], a.children[2,1], y[1:a.children[1,1].m], -1.0)
                hmat_solve!(a.children[2,2], y[a.children[1,1].m+1:end], true)
            end
        end
    else
        if a.is_fullmatrix
            LAPACK.trtrs!('U', 'N', 'N', a.C, y)
        elseif a.is_hmat
            @views begin
                hmat_solve!(a.children[2,2], y[a.children[1,1].m+1:end], false)
                hmat_matvec!(y[1:a.children[1,1].m], a.children[1,2], y[a.children[1,1].m+1:end], -1.0)
                hmat_solve!(a.children[1,1], y[1:a.children[1,1].m], false)
            end
        end
    end
end


# given a factorized matrix a, solve a x = y
function Base.:\(a::Hmat, y::AbstractArray{Float64})
    if length(a.P)==0
        @error "a has not been factorized"
    end
    w = deepcopy(y)
    permute!(w, a.P)
    hmat_solve!(a, w, true)
    hmat_solve!(a, w, false)
    return w
end

function solve(a::Hmat, y::AbstractArray{Float64}, P::Int64, Q::Int64)
    w = deepcopy(y)
    w = w[P]
    w = a\w
    w = w[Q]
    return w
end

function color_level(H)
    function helper!(H, level)
        if H.is_fullmatrix
            H.C = ones(size(H.C))* (rand()*0.5)
        elseif H.is_rkmatrix
            H.A = -ones(H.m, 1)
            H.B = ones(H.n, 1) * (level + rand()*0.5)
        else
            for i = 1:2
                for j = 1:2
                    helper!(H.children[i,j], level+1)
                end
            end
        end
    end
    helper!(H, 0)
    to_fmat!(H)
    return H.C
end

function plot_hmat(H)
    C = color_level(H)
    matshow(C)
end

function PyPlot.:matshow(H::Hmat)
    P = copy(H)
    C = color_level(P)
    matshow(C)
end

function printmat(H)
    println("=============== size = $(size(H,1))x$(size(H,2)) ===================")
    for i = 1:size(H,1)
        for j = 1:size(H,2)
            @printf("%+0.4f ", H[i,j])
        end
        @printf("\n")
    end
    println("=====================================================================")
end

function check_if_equal(H::Hmat, C::Array{Float64})
    G = (H)
    to_fmat!(G)
    println("Error = $(norm(C-G.C,2)/norm(C,2))")
end

function verify_matrix_error(H::Hmat, C::Array{Float64})
    G = to_fmat(H)
    err = pointwise_error(G, C)
    println("Matrix Error = $err")
end

function verify_matvec_error(H::Hmat, C::Array{Float64})
    y = rand(size(C,1))
    b1 = H*y
    b2 = C*y
    err = norm(b2-b1)/norm(b2)
    println("Matvec Error = $err")
end

function verify_lu_error(HH::Hmat; A = nothing)
    H = copy(HH)
    C = to_fmat(H)
    lu!(H)
    x = rand(size(C,1))
    b = C*x
    y = H\b
    err = norm(x-y)/norm(x)
    # println("Permuation = $(H.P)")
    println("Solve Error = $err")
    to_fmat!(H)
    G = C[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
    println("LU Operator Error = $(maximum(abs.(G)))")

    
    if A!=nothing
        G = A[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
        println("LU Matrix Error = $(maximum(abs.(G)))")
        x = rand(size(A,2))
        b = A*x
        y = H\b
        err = norm(x-y)/norm(x)
        # println("Permuation = $(H.P)")
        println("Matrix Solve Error = $err")
    end
    return G
end

function check_err(HH::Hmat, C::Array{Float64})
    H = copy(HH)
    to_fmat!(H)
    # println(H.P)
    G = C[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
    println("Matrix Error = $(maximum(abs.(G)))")
end

function pointwise_error(A, B)
    return maximum(abs.(A-B))
end

function rel_error(x::Array{Float64}, y::Array{Float64})
    return norm(y-x)/norm(y)
end