import LowRankApprox: psvd
export 
lu!,
lu,
getl,
getu,
aca2

function aca(A::Array{Float64}, Rrank::Int64)
    U = zeros(size(A,1), Rrank+1)
    V = zeros(size(A,2), Rrank+1)
    R = ccall((:aca_wrapper,"deps/src/build/libaca"), Cint, (Ref{Cdouble}, Ref{Cdouble},
                Ref{Cdouble},Cint, Cint,Cdouble, Cint ), A, U, V, size(A,1), size(A,2), Hparams.εComp, Rrank)
    R = min(R, Rrank+1)
    U = U[:,1:R]
    V = V[:,1:R]
    return U, V
end

function bbfmm(f::Function, X::Array{Float64},Y::Array{Float64}, Rrank::Int64)
    if length(size(X))==1
        return bbfmm1d(f, X, Y, Rrank)
    else
        if Rrank^2>max(size(X,1), size(Y,1))
            Rrank=Int64(floor(sqrt(max(size(X,1), size(Y,1)))))
        end
        return bbfmm2d(f, X, Y, Rrank)
    end
end

function aca2(f::Function, X::Array{Float64,1}, Y::Array{Float64,1}, Rrank::Int64)
    m, n = length(X),length(Y)
    τr = 1
    A = zeros(m, Rrank+1)
    B = zeros(n, Rrank+1)
    Ib = ones(Bool, n)
    for r = 0:Rrank
        Mτr = [f(X[τr], y) for y in Y]
        σr = argmax(abs.(Mτr))
        Ib[σr] = false
        δ = Mτr[σr] - B[σr,:]'*A[τr,:]
        if δ == 0
            return A[:,1:r], B[:,1:r]
        else
            Mσr = [f(x, Y[σr]) for x in X]
            A[:,r+1] = (Mσr-(A[:,1:r]*B[σr,1:r])[:])/δ
            B[:,r+1] = Mτr-(A[τr,1:r]'*B[:,1:r]')[:]
        end
        τr = argmax(abs.(B[Ib,r+1]))
        τr = findall(Ib)[τr]
        ε = norm(A[:,r+1])*norm(B[:,r+1])/norm(A[:,1])/norm(B[:,1])
        if ε<=Hparams.εComp
            return A[:,1:r+1], B[:,1:r+1]
        end
    end
    return A, B
end

# reference: https://www.theses.fr/2016GREAT120.pdf
function aca2(f::Function, X::Array{Float64,2}, Y::Array{Float64,2}, Rrank::Int64)
    local σr = nothing
    local δ = nothing
    local Mτr = nothing

    m, n = size(X, 1),size(Y, 1)
    τr = 1
    A = zeros(m, Rrank+1)
    B = zeros(n, Rrank+1)
    Ia = ones(Bool, m)
    patience = 3
    for r = 0:Rrank
        for i = 1:patience
            Mτr = [f(X[τr,:], Y[i,:]) for i=1:n]
            if r>0
                Mτr = Mτr-B[:,1:r]*A[τr,1:r]
            end
            σr = argmax(abs.(Mτr))
            
            δ = Mτr[σr]
            if δ==0 || i==patience
                break
            else
                τr = rand(findall(Ia)) # choose a new pivot
            end
        end

        if δ==0
            return A[:,1:r], B[:,1:r]
        end
        Ia[τr] = false
        
        # update A, B
        Mσr = [f(X[i,:], Y[σr,:]) for i=1:m]
        if r==0
            A[:,r+1] = Mσr/δ
        else
            A[:,r+1] = (Mσr-(A[:,1:r]*B[σr,1:r])[:])/δ
        end
        B[:,r+1] = Mτr
    
        # generate a new τr
        τr = argmax(abs.(A[Ia,r+1]))
        τr = findall(Ia)[τr]
        ε = norm(A[:,r+1])*norm(B[:,r+1])/norm(A[:,1])/norm(B[:,1])
        if ε<=Hparams.εComp
            return A[:,1:r+1], B[:,1:r+1]
        end
    end
    # @show maximum(abs.(A*B'-FullMat(f, X, Y)))
    return A, B
end

function bbfmm1d(f::Function, X::Array{Float64},Y::Array{Float64}, Rrank::Int64)
    U = zeros(size(X,1), Rrank)
    V = zeros(size(Y,1), Rrank)
    f_c = @cfunction($f, Cdouble, (Cdouble, Cdouble));
    ccall((:bbfmm1D,"deps/src/build/libbbfmm"), Cvoid,
            (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Cint, Cint, Cint),
            f_c, X, Y, minimum(X), maximum(X) ,minimum(Y), maximum(Y) ,U,V, Rrank, length(X), length(Y))
    return U, V
end

function bbfmm2d(f::Function, X::Array{Float64,2},Y::Array{Float64,2}, Rrank::Int64)
    U = zeros(size(X,1), Rrank^2)
    V = zeros(size(Y,1), Rrank^2)
    g = (x1,y1,x2,y2)->f([x1;y1],[x2;y2])
    f_c = @cfunction($g, Cdouble, (Cdouble, Cdouble, Cdouble, Cdouble));
    ccall((:bbfmm2D,"deps/src/build/libbbfmm"), Cvoid,
            (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cint, Cint, Cint),
            f_c, X[:], Y[:],U,V, Rrank, size(X,1), size(Y,1))
    return U, V
end

# column pivoting RRQR
function rrqr(A::Array{Float64})
    F = pqrfact(A, rtol = Hparams.εComp)
    ip = inverse_permutation(F.p)
    Q = F.Q
    R = F.R[:,ip]'
    return Q, R
end


function rank_truncate(S::Array{Float64})
    if length(S)==0
        return 0
    end
    k = findlast(S/S[1] .> Hparams.εTrunc)
    if isa(k, Nothing)
        return 0
    else
        return k
    end
end

# C is a full matrix
# the function will try to compress C with given tolerance eps
# Rrank is required when method = "aca"
function compress(C::Array{Float64})
    method = Hparams.CompMethod
    # method = "rrqr"
    # if the matrix is a zero matrix, return zero vectors
    if sum(abs.(C))<1e-5
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
        k = findlast(S/S[1] .> Hparams.εTrunc)
        @assert !isa(k, Nothing) # k should never be zero
        A = U[:,1:k]
        B = (diagm(0=>S[1:k])*V'[1:k,:])'
        return A, B
    elseif method=="aca"             # more accurate
        U,V = aca(C, Hparams.εComp, Hparams.MaxRank);
        return U,V
    elseif method=="rrqr"
        U,V = rrqr(C)
        return U,V
    else
        error("Method $method not implemented yet")
    end

end


function getl(A, unitdiag=true)
    if unitdiag
        return LowerTriangular(A)+LowerTriangular(-diagm(0=>diag(A)) + UniformScaling(1.0))
    else
        return LowerTriangular(A)
    end
end

function getu(A, unitdiag=false)
    if unitdiag
        return UpperTriangular(A)+UpperTriangular(-diagm(0=>diag(A)) + UniformScaling(1.0))
    else
        return UpperTriangular(A)
    end
end

# function transpose!(a::Hmat)
#     a.m, a.n = a.n, a.m
#     a.s, a.t = a.t, a.s
#     if a.is_rkmatrix
#         a.A, a.B = a.B, a.A
#     elseif a.is_fullmatrix
#         # a.C = a.C'
#         @timeit tos "b2" a.C = a.C'   # bottleneck
#     else
#         for i = 1:2
#             for j = 1:2
#                 transpose!(a.children[i,j])
#             end
#         end
#         a.children[1,2], a.children[2,1] = a.children[2,1], a.children[1,2]
#     end
# end

########################### LU related functions ###########################

# # special function for computing c = c - a*b
function hmat_sub_mul!(c::Hmat, a::Hmat, b::Hmat)
    M = hmul(a, b)
    hmat_add!(c, M , -1.0)
end

# solve a x = b where a is possibly a H-matrix. a is lower triangular. 
function mat_full_solve(a::Hmat, b::AbstractArray{Float64}, unitdiag::Bool, ul::Char='L')
    if a.is_rkmatrix
        error("A should not be a low-rank matrix")
    end
    if unitdiag
        cc = 'U'
    else
        cc = 'N'
    end

    if ul == 'L'
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
            mat_full_solve(a11, b1, unitdiag, ul)
            b2 -= a21 * b1
            # hmat_sub_mul!(b2, a21, b1)
            mat_full_solve(a22, b2, unitdiag, ul)
            # b[:] = [b1;b2]
            b[1:n,:] = b1
            b[n+1:end,:] = b2
            # println("Error = ", maximum(abs.(b-x)))
        end
    else
        if a.is_fullmatrix
            # p = b/getu(to_fmat(a), unitdiag)
            c = Array(b')
            LAPACK.trtrs!('U', 'T', cc, a.C, c)
            b[:] = c'
            # @show size(b), size(p)
            # println("**** trtrs error = ", pointwise_error(b, p))
        elseif a.is_hmat
            # p = b/getu(to_fmat(a), unitdiag)
            a11 = a.children[1,1]
            a12 = a.children[1,2]
            a22 = a.children[2,2]
            n = a11.m
            # @show size(a11), size(b), size(a)
            b1 = b[:,1:n]
            b2 = b[:,n+1:end]
            mat_full_solve(a11, b1, unitdiag, ul)
            # b1 = b1/getu(to_fmat(a11), unitdiag)
            b2 -= b1 * a12
            mat_full_solve(a22, b2, unitdiag, ul)
            # b2 = b2/getu(to_fmat(a22), unitdiag)
            b[:,1:n] = b1
            b[:,n+1:end] = b2
            # println("**** mat_full_solve error = ", pointwise_error(b, p))
        end
    end
end

# Solve AX = B and store the result into B
# A, B have been prepermuted and therefore this function should not worry about permutation
function hmat_trisolve!(a::Hmat, b::Hmat, islower::Bool, unitdiag::Bool)
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
        # @show size(a), size(b)
        # @assert a.n==b.n
        # println("Upper Triangular")
        if a.is_fullmatrix && b.is_fullmatrix
            # println("Here")
            # p =b.C/ getu(to_fmat(a), unitdiag)
            b.C = b.C'
            LAPACK.trtrs!('U', 'T', cc, a.C, b.C)
            b.C = b.C'
            # println("*** trtrs Error = ", pointwise_error(p, to_fmat(b)))
        elseif a.is_fullmatrix && b.is_rkmatrix
            if size(b.A,1)==0
                @warn "b is an empty matrix"
                return
            end
            b.B = b.B'
            LAPACK.trtrs!('U', 'T', cc, a.C, b.B)
            b.B = b.B'
        elseif a.is_hmat && b.is_fullmatrix
            # println("HF")
            # p =b.C/ getu(to_fmat(a), unitdiag)
            # error("To be implemented")
            mat_full_solve(a, b.C, unitdiag, 'U')
            # println("*** HF Error = ", pointwise_error(p, to_fmat(b)))
        elseif a.is_hmat && b.is_rkmatrix
            # println("FH")
            # p = to_fmat(b)/getu(to_fmat(a), unitdiag)
            # error("To be implemented")
            mat_full_solve(a, b.B', unitdiag, 'U')
            # println("*** FH Error = ", pointwise_error(p, to_fmat(b)))
            # error("Not used")

        elseif a.is_hmat && b.is_hmat
            # println("HH")
            # p = to_fmat(b)/getu(to_fmat(a), unitdiag)
            a11, a12, a21, a22 = a.children[1,1], a.children[1,2],a.children[2,1],a.children[2,2]
            b11, b12, b21, b22 = b.children[1,1], b.children[1,2],b.children[2,1],b.children[2,2]
            
            # p = to_fmat(b11)/getu(to_fmat(a11), unitdiag)
            hmat_trisolve!(a11, b11, islower, unitdiag)
            # println("*** I Error = ", pointwise_error(p, to_fmat(b11)))

            # p = getl(to_fmat(a11), unitdiag)\to_fmat(b12)
            hmat_trisolve!(a11, b21, islower, unitdiag)
            # println("*** II Error = ", pointwise_error(p, to_fmat(b12)))

            # p = to_fmat(b21)-to_fmat(a21)*to_fmat(b11)
            hmat_sub_mul!(b12, b11, a12)
            # println("*** III Error = ", pointwise_error(p, to_fmat(b21)))

            # p = to_fmat(b22)-to_fmat(a21)*to_fmat(b12)
            hmat_sub_mul!(b22, b21, a12)
            # println("*** IV Error = ", pointwise_error(p, to_fmat(b22)))

            # p = getl(to_fmat(a22), unitdiag)\to_fmat(b21)
            hmat_trisolve!(a22, b12, islower, unitdiag)
            # println("*** V Error = ", pointwise_error(p, to_fmat(b21)))

            # p = getl(to_fmat(a22), unitdiag)\to_fmat(b22)
            hmat_trisolve!(a22, b22, islower, unitdiag)
            # println("*** VI Error = ", pointwise_error(p, to_fmat(b22)))

            # println("*** H Error = ", pointwise_error(p, to_fmat(b)))
        
        else
            error("Invalid")
        end

        # @timeit tos "t" transpose!(a)
        # @timeit tos "t" transpose!(b)
        # hmat_trisolve!(a, b, true, unitdiag)
        # @timeit tos "t" transpose!(a)
        # @timeit tos "t" transpose!(b)
    end
end

# Note: we do not need to permute the cluster. The only information we will use is the 
# total number of nodes instead of relative order of the points
function permute_hmat!(H::Hmat, P::AbstractArray{Int64})
    if H.is_fullmatrix
        H.C = H.C[P,:] # bottleneck
        # @inbounds for i = 1:size(H.C,2)
        #     @views permute!(H.C[:,i], P)
        # end
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
    if length(H.P)>0
        @warn "H-matrix H is already factorized; reuse factorization"
        return 
    end
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
        H.P = [H.children[1,1].P; H.children[2,2].P .+ H.children[1,1].m];
        # GG = to_fmat(H)
        # println("***", size(G), maximum(abs.(G-GG)))
    end
    return H
end

function LinearAlgebra.:lu(H::Hmat)
    A = Hmat()
    hmat_copy!(A, H)
    return lu!(A)
end

# a is factorized hmatrix
function hmat_solve!(a::Hmat, y::AbstractArray{Float64}, lower::Bool=true)
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