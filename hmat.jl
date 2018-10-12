using Parameters
using LinearAlgebra
using PyPlot
using Printf

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
    if k==nothing
        k = 0
    end
    return k
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
    C = A1*B1' + A2*B2'
    return compress(C, 1e-6, Inf)
end


function rkmat_add!(a, b, scalar, method=1)
    
    if method==1
        a.A, a.B = svdtrunc(a.A, a.B, scalar*b.A, b.B)
    else
        error("Method not defined!")
    end
end

# Perform a = a + b
function hmat_add!( a, b, scalar = 1.0)
    if a.is_fullmatrix && b.is_fullmatrix
        a.C += scalar * b.C
    elseif a.is_fullmatrix && b.is_rkmatrix
        a.C += scalar * b.A * b.B'
    elseif a.is_fullmatrix && b.is_hmat
        c = Hmat()
        hmat_copy!(c, b)
        to_fmat!(c)
        a.C += c.C
    elseif a.is_rkmatrix && b.is_fullmatrix
        to_fmat!(a)
        hmat_add!(a, b, scalar)
    elseif a.is_rkmatrix && b.is_rkmatrix
        rkmat_add!(a, b, scalar, 1)
    elseif a.is_rkmatrix && b.is_hmat
        to_fmat!(a)
        hmat_add!(a, b, scalar)
    elseif a.is_hmat && b.is_fullmatrix
        to_fmat!(a)
        hmat_add!(a, b, scalar)
    elseif a.is_hmat && b.is_rkmatrix
        to_fmat!(a)
        hmat_add!(a, b, scalar)
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
    # consistency(c)
    hmat_add!( c, b, 1.0)
    return c
end

function Base.:*(a::Hmat, b::Hmat)
    H = Hmat(m=a.m, n = b.n)
    if a.is_fullmatrix && b.is_fullmatrix
        H.C = a.C * b.C
        H.is_fullmatrix = true
    elseif a.is_fullmatrix && b.is_rkmatrix
        H.C = (a.C * b.A) * b.B'
        H.is_fullmatrix = true
    elseif a.is_fullmatrix && b.is_hmat
        H.is_fullmatrix = true
        c = Hmat()
        hmat_copy!(c, b)
        to_fmat!(c)
        H.C = a*c
    elseif a.is_rkmatrix && b.is_fullmatrix
        H.is_rkmatrix = true
        H.A = a.A 
        H.B = (a.B' * b.C)'
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
    elseif a.is_hmat && b.is_fullmatrix
        H.is_fullmatrix = true
        c = Hmat()
        hmat_copy!(c, a)
        to_fmat!(c)
        H.C = c.C*b.C
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

function Base.:*(a::Hmat, v::AbstractArray{Float64})
    if a.is_fullmatrix
        return a.C*v
    elseif a.is_rkmatrix
        return a.A * (a.B'*v)
    else
        u = zeros(length(v))
        m, n = a.children[1,1].m, a.children[1,1].n
        @views begin
            u[1:m] = a.children[1,1]*v[1:n] + a.children[1,2]*v[n+1:end]
            u[m+1:end] = a.children[2,1]*v[1:n] + a.children[2,2]*v[n+1:end]
        end
        return u
    end
end

function hmat_matvec!(r::AbstractArray{Float64}, a::Hmat, v::Array{Float64})
    if a.is_fullmatrix
        r[:] += a.C * v
    elseif a.is_rkmatrix
        r[:] += a.A * (a.B'*v)
    else
        m, n = a.children[1,1].m, a.children[1,1].n
        @views begin
            hmat_matvec!(r[1:m], a.children[1,1], v[1:n])
            hmat_matvec!(r[1:m], a.children[1,2], v[1:n])
            hmat_matvec!(r[m+1:end], a.children[2,1], v[n+1:end])
            hmat_matvec!(r[m+1:end], a.children[2,2], v[n+1:end])
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

    if islower
        if a.is_fullmatrix && b.is_fullmatrix
            if permutation && length(a.P)>0
                b.C = b.C[a.P,:]
            end
            b.C = getl(a.C, unitdiag)\b.C
        elseif a.is_fullmatrix && b.is_rkmatrix
            if permutation && length(a.P)>0
                b.A = b.A[a.P,:]
            end
            if size(b.A,1)==0
                return
            end
            b.A = getl(a.C, unitdiag)\b.A
        elseif a.is_fullmatrix && b.is_hmat
            b.is_fullmatrix = true
            to_fmat!(b)
            hmat_trisolve!(a, b, islower, unitdiag, permutation)
        elseif a.is_hmat && b.is_hmat
            hmat_trisolve!(a.children[1,1], b.children[1,1], islower, unitdiag, permutation)
            hmat_trisolve!(a.children[1,1], b.children[1,2], islower, unitdiag, permutation)
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
            hmat_trisolve!(H, b, islower, unitdiag, permutation)
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
        print("LU ")
        @time L, U, H.P = lu(H.C)
        H.C = L - UniformScaling(1.0) + U
    else
        lu!(H.children[1,1])
        print("L ")
        @time hmat_trisolve!(H.children[1,1], H.children[1,2], true, true, true)
        print("U ")
        @time hmat_trisolve!(H.children[1,1], H.children[2,1], false, false, false)
        print("+ ")
        @time hmat_add!(H.children[2,2], H.children[2,1]*H.children[1,2], -1.0)
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
            ldiv!(getl(a.C, true), y)
            # y[:] = getl(a.C, true)\y
        elseif a.is_hmat
            k = a.children[1,1].m
            # println("$k, $(length(y))")
            @views begin
                hmat_solve!(a.children[1,1], y[1:k], true)
                y[k+1:end] -= a.children[2,1]*y[1:k]
                hmat_solve!(a.children[2,2], y[k+1:end], true)
            end
        end
    else
        if a.is_fullmatrix
            ldiv!(getu(a.C,false), y)
            # y = getu(a.C, false)\y
        elseif a.is_hmat
            k = a.children[1,1].m
            @views begin
                hmat_solve!(a.children[2,2], y[k+1:end], false)
                y[1:k] -= a.children[1,2]*y[k+1:end]
                hmat_solve!(a.children[1,1], y[1:k], false)
            end
        end
    end
end

# a is factorized hmatrix
function LinearAlgebra.:ldiv!(a::Hmat, y::AbstractArray{Float64})
    hmat_solve!(a, y, true)
    hmat_solve!(a, y, false)
end

function Base.:\(a::Hmat, y::AbstractArray{Float64})
    w = copy(y)
    ldiv!(a, w)
    return w
end


############### algebraic constructor ##################
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
            H.B = (diagm(0=>S[1:k])*V'[1:k,:])'
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



