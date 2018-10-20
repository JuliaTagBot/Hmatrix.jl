using Parameters
using LinearAlgebra
using PyPlot
using Printf
using Statistics
using Profile
using LowRankApprox

if !@isdefined Hmat
    @with_kw mutable struct Hmat
        A::Array{Float64} = Array{Float64}([])
        B::Array{Float64} = Array{Float64}([])
        C::Array{Float64} = Array{Float64}([])
        P::Array{Int64} = Array{Int64}([])
        is_rkmatrix::Bool = false
        is_fullmatrix::Bool = false
        is_hmat::Bool = false
        m::Int = 0
        n::Int = 0
        children::Array{Hmat} = Array{Hmat}([])
    end
end

# utilities
function consistency(H, L=@__LINE__)
    try
        if H.m==0 || H.n==0
            @assert false
        end
        if H.is_rkmatrix
            @assert size(H.A,1)==H.m
            @assert size(H.B,1)==H.n
        elseif H.is_fullmatrix
            if !(size(H.C,1)==H.m && size(H.C,2)==H.n)
                error("$(size(H.C))!=[$(H.m), $(H.n)]")
            end
        elseif H.is_hmat
            n1 = Int(round(H.n/2))
            m1 = Int(round(H.m/2))
            size(H.children[1,1].m)==m1
            size(H.children[1,1].n)==m1
            size(H.children[1,2].m)==m1
            size(H.children[1,2].n)==H.n-n1
            size(H.children[2,1].m)==H.m-m1
            size(H.children[2,1].n)==n1
            size(H.children[2,2].m)==H.m-m1
            size(H.children[2,2].n)==H.n-n1
            for i = 1:2
                for j = 1:2
                    consistency(H.children[i,j])
                end
            end
        end
    catch
        println("Assertion: $L")
        println(H)
        @assert false
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


function fmat(A::Array{Float64})
    H = Hmat()
    H.is_fullmatrix = true
    H.m, H.n = size(A)
    H.C = copy(A)
    return H
end

function rkmat(A, B)
    H = Hmat()
    H.is_rkmatrix = true
    m = size(A,1)
    n = size(B,1)
    H.A = copy(A)
    H.B = copy(B)
    return H
end

function rank_truncate(S, eps=1e-6)
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

function compress(C, eps=1e-6, N = nothing)
    U,S,V = svd(C)
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

function svdtrunc(A1, B1, A2, B2)
    # C = A1*B1' + A2*B2'
    # return compress(C, 1e-6, Inf)

    FA = qr([A1 A2])
    FB = qr([B1 B2])
    U,S,V = svd(FA.R*FB.R')
    k = rank_truncate(S, 1e-6)
    A = FA.Q * U[:,1:k] * diagm(0=>S[1:k])
    B = FB.Q * V[:,1:k]
    return A, B
end


function rkmat_add!(a, b, scalar, method=1)

    if method==1
        a.A, a.B = svdtrunc(a.A, a.B, scalar*b.A, b.B)
        # svdtrunc!(a, b)
    else
        error("Method not defined!")
    end
end

function hmat_full_add!(a::Hmat, b::Array{Float64}, scalar)
    if a.is_fullmatrix
        a.C += b*scalar
    elseif a.is_rkmatrix
        # to_fmat!(a)
        # a.C += scalar * b
        C = a.A*a.B'+scalar*b
        a.A, a.B = compress(C)
    elseif a.is_hmat
        m = a.children[1,1].m
        hmat_full_add!(a.children[1,1], b[1:m,1:m],scalar)
        hmat_full_add!(a.children[1,2], b[1:m,m+1:end],scalar)
        hmat_full_add!(a.children[2,1], b[m+1:end,1:m],scalar)
        hmat_full_add!(a.children[2,2], b[m+1:end,m+1:end],scalar)
    else
        error("Should not be here")
    end
end
# Perform a = a + b
function hmat_add!( a, b, scalar = 1.0)
    if b.is_fullmatrix
        hmat_full_add!(a, b.C, scalar)
    elseif a.is_fullmatrix && b.is_rkmatrix
        a.C += scalar * b.A * b.B'
    elseif a.is_fullmatrix && b.is_hmat
        c = Hmat()
        hmat_copy!(c, b)
        to_fmat!(c)
        a.C += c.C
    elseif a.is_rkmatrix && b.is_rkmatrix
        rkmat_add!(a, b, scalar, 1)
    elseif a.is_rkmatrix && b.is_hmat
        # this is never used
        to_fmat!(a)
        hmat_add!(a, b, scalar)
    elseif a.is_hmat && b.is_rkmatrix
        # to_fmat!(a)
        # hmat_add!(a, b, scalar)
        hmat_full_add!(a, b.A*b.B', scalar)
    elseif a.is_hmat && b.is_hmat
        for i = 1:2
            for j = 1:2
                hmat_add!(a.children[i,j], b.children[i,j], scalar)
            end
        end
    end
end

function Base.:+(a::Hmat, b::Hmat)
    c = Hmat()
    hmat_copy!(c, a)
    hmat_add!( c, b, 1.0)
    return c
end

function full_mat_mul(a::Array{Float64, 2}, b::Hmat)
    H = Hmat(m=size(a,1), n = b.n)
    
    if b.is_hmat
        m, n = b.children[1,1].m, b.children[1,1].n
        p, q = b.m - m, b.n - n
        H.is_hmat = true
        H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
        H.children[1,1] = full_mat_mul(a[1:n, 1:m] , b.children[1,1]) + full_mat_mul(a[1:n, m+1:m+p] , b.children[1,2])
        H.children[1,2] = full_mat_mul(a[1:n, 1:m] , b.children[2,1]) + full_mat_mul(a[1:n, m+1:m+p] , b.children[2,2])
        H.children[2,1] = full_mat_mul(a[n+1:n+q, 1:m] , b.children[1,1]) + full_mat_mul(a[n+1:n+q, m+1:m+p] , b.children[1,2])
        H.children[2,2] = full_mat_mul(a[n+1:n+q, 1:m] , b.children[1,2] ) + full_mat_mul(a[n+1:n+q, m+1:m+p] , b.children[2,2])
    elseif b.is_fullmatrix
        H.is_fullmatrix = true
        H.C = a * b.C
    else
        H.is_rkmatrix = true
        H.A = a * b.A
        H.B = b.B
    end
    return H
end

function mat_full_mul(a::Hmat, b::Array{Float64, 2})
    H = Hmat(m=a.m, n = size(b,2))
    
    if a.is_hmat
        m, n = a.children[1,1].m, a.children[1,1].n
        p, q = a.m - m, a.n - n
        H.is_hmat = true
        H.children = Array{Hmat}([Hmat() Hmat()
                                    Hmat() Hmat()])
        b11 = b[1:n, 1:m]
        b12 = b[1:n, m+1:m+p]
        b21 = b[n+1:n+q, 1:m]
        b22 = b[n+1:n+q, m+1:m+p]
        a11 = a.children[1,1]
        a12 = a.children[1,2]
        a21 = a.children[2,1]
        a22 = a.children[2,2]
        H.children[1,1] = mat_full_mul(a11, b11) + mat_full_mul(a12, b21) 
        H.children[1,2] = mat_full_mul(a11, b12) + mat_full_mul(a12, b22)
        H.children[2,1] = mat_full_mul(a21, b11) + mat_full_mul(a22, b21)
        H.children[2,2] = mat_full_mul(a21, b12) + mat_full_mul(a22, b22)
    elseif a.is_fullmatrix
        H.is_fullmatrix = true
        H.C = a.C * b
    else
        H.is_rkmatrix = true
        H.A = a.A
        H.B = b' * a.B
    end
    return H
end

function Base.:*(a::Hmat, b::Hmat)
    H = Hmat(m=a.m, n = b.n)
    if a.is_fullmatrix
        H = full_mat_mul(a.C, b)
    elseif b.is_fullmatrix
        H = mat_full_mul(a, b.C)
    elseif a.is_rkmatrix && b.is_rkmatrix
        H.is_rkmatrix = true
        H.A = a.A
        H.B = b.B * (b.A' * a.B)
    elseif a.is_rkmatrix && b.is_hmat
        H.is_rkmatrix = true
        c = Hmat()
        hmat_copy!(c, b)
        to_fmat!(c)
        H.A = a.A
        H.B =  c.C' * a.B
    
    elseif a.is_hmat && b.is_rkmatrix
        H.is_rkmatrix = true
        c = Hmat()
        hmat_copy!(c, a)
        to_fmat!(c)
        H.A = c*b.A
        H.B = b.B
    elseif a.is_hmat && b.is_hmat
        H.is_hmat = true
        H.children = Array{Hmat}([Hmat() Hmat()
                                 Hmat() Hmat()])
        for i = 1:2
            for j = 1:2
                H.children[i,j] = a.children[i,1]*b.children[1,j] + a.children[i,2]*b.children[2,j]
            end
        end
    end
    return H
end

# function Base.:*(a::Hmat, v::AbstractArray{Float64})
#     if a.is_fullmatrix
#         return a.C*v
#     elseif a.is_rkmatrix
#         return a.A * (a.B'*v)
#     else
#         u = zeros(length(v))
#         m, n = a.children[1,1].m, a.children[1,1].n
#         @views begin
#             u[1:m] = a.children[1,1]*v[1:n] + a.children[1,2]*v[n+1:end]
#             u[m+1:end] = a.children[2,1]*v[1:n] + a.children[2,2]*v[n+1:end]
#         end
#         return u
#     end
# end

function Base.:*(a::Hmat, v::AbstractArray{Float64})
    r = zeros(size(v))
    for i = 1:size(v,2)
        @views hmat_matvec!(r[:,i], a, v[:,i], 1.0)
    end
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
function hmat_copy!(H, A)
    H.m = A.m
    H.n = A.n
    if A.is_fullmatrix
        H.C = copy(A.C)
        H.P = copy(A.P)
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

# convert matrix A to full matrix
function to_fmat!(A::Hmat)
    if A.is_fullmatrix
        return
    elseif A.is_rkmatrix
        A.C = A.A*A.B'
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
    end
    A.is_fullmatrix = true
    A.is_rkmatrix = false
    A.is_hmat = false
end

function to_fmat2!(A::Hmat)
    if length(A.C)>0
        return
    end
    if A.is_fullmatrix
        return
    elseif A.is_rkmatrix
        A.C = A.A*A.B'
    elseif A.is_hmat
        for i = 1:2
            for j = 1:2
                to_fmat2!(A.children[i,j])
            end
        end
        A.C = [A.children[1,1].C A.children[1,2].C
                A.children[2,1].C  A.children[2,2].C]
        if length(A.children[1,1].P)>0
            A.P = [A.children[1,1].P;
                    A.children[2,2].P .+ A.children[1,1].m]
        end
    end
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


# Solve AX = B and store the result into B
function hmat_trisolve!(a::Hmat, b::Hmat, islower, unitdiag, permutation)
    if a.is_rkmatrix
        error("A should not be a low-rank matrix")
    end

    if unitdiag
        cc = 'U'
    else
        cc = 'N'
    end

    if islower
        if a.is_fullmatrix && b.is_fullmatrix
            if permutation && length(a.P)>0
                b.C = b.C[a.P,:]
            end
            LAPACK.trtrs!('L', 'N', cc, a.C, b.C)
        elseif a.is_fullmatrix && b.is_rkmatrix
            if permutation && length(a.P)>0
                b.A = b.A[a.P,:]
            end
            if size(b.A,1)==0
                return
            end
            LAPACK.trtrs!('L', 'N', cc, a.C, b.A)
        elseif a.is_fullmatrix && b.is_hmat
            b.is_fullmatrix = true
            to_fmat!(b)
            if permutation && length(a.P)>0
                b.C = b.C[a.P,:]
            end
            LAPACK.trtrs!('L', 'N', cc, a.C, b.C)

        elseif a.is_hmat && b.is_hmat
            hmat_trisolve!(a.children[1,1], b.children[1,1], islower, unitdiag, permutation)
            hmat_trisolve!(a.children[1,1], b.children[1,2], islower, unitdiag, permutation)
            # hmat_matvec!(b.children[2,1], a.children[2,1], b.children[1,1], -1.0)
            # hmat_matvec!(b.children[2,2], a.children[2,1], b.children[1,2], -1.0)
            hmat_add!(b.children[2,1], a.children[2,1]*b.children[1,1], -1.0)
            hmat_add!(b.children[2,2], a.children[2,1]*b.children[1,2], -1.0)
            hmat_trisolve!(a.children[2,2], b.children[2,1], islower, unitdiag, permutation)
            hmat_trisolve!(a.children[2,2], b.children[2,2], islower, unitdiag, permutation)
        elseif a.is_hmat && b.is_fullmatrix
            H = Hmat()
            hmat_copy!(H, a)
            to_fmat!(H)
            hmat_trisolve!(H, b, islower, unitdiag, permutation)
        elseif a.is_hmat && b.is_rkmatrix
            H = Hmat()
            hmat_copy!(H, a)
            to_fmat!(H)
            if permutation && length(a.P)>0
                b.A = b.A[a.P,:]
            end
            LAPACK.trtrs!('L', 'N', cc, H.C, b.A)
        end
    else
        transpose!(a)
        transpose!(b)
        hmat_trisolve!(a, b, true, unitdiag, permutation)
        transpose!(a)
        transpose!(b)
    end
end

function LinearAlgebra.:lu!(H::Hmat)
    if H.is_rkmatrix
        error("H should not be a low-rank matrix")
    end

    if H.is_fullmatrix
        F = lu!(H.C)
        H.P = F.p
    else
        lu!(H.children[1,1])
        hmat_trisolve!(H.children[1,1], H.children[1,2], true, true, true)
        hmat_trisolve!(H.children[1,1], H.children[2,1], false, false, false)
        hmat_add!(H.children[2,2], H.children[2,1]*H.children[1,2], -1.0)
        lu!(H.children[2,2])
    end
end

# a is factorized hmatrix
function hmat_solve!(a::Hmat, y::AbstractArray{Float64}, lower=true)
    if a.is_rkmatrix
        error("a cannot be a low-rank matrix")
    end
    if lower
        if a.is_fullmatrix
            permute!(y, a.P)
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

# a is factorized hmatrix
# these implementations makes H a preconditioner
function LinearAlgebra.:ldiv!(x::AbstractArray{Float64}, a::Hmat, y::AbstractArray{Float64})
    x = deepcopy(y)
    hmat_solve!(a, y, true)
    hmat_solve!(a, y, false)
end

function LinearAlgebra.:ldiv!(a::Hmat, y::AbstractArray{Float64})
    hmat_solve!(a, y, true)
    hmat_solve!(a, y, false)
end

function Base.:\(a::Hmat, y::AbstractArray{Float64})
    w = deepcopy(y)
    ldiv!(a, w)
    return w
end

function color_level(H)
    function helper!(H, level)
        if H.is_fullmatrix
            H.C = ones(size(H.C))* (-rand()*0.8)
        elseif H.is_rkmatrix
            H.A = ones(H.m, 1)
            H.B = ones(H.n, 1) * (level + rand()*0.8)
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
    P = Hmat()
    hmat_copy!(P, H)
    C = color_level(P)
    matshow(C)
end

function check_if_equal(H::Hmat, C::Array{Float64})
    G = Hmat()
    hmat_copy!(G, H)
    to_fmat!(G)
        println("Error = $(norm(C-G.C,2)/norm(C,2))")
end