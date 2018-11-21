# harithm.jl
# This file contains the arithmetic functions for H-matrix


# a = a + scalar * b
# a is a H-matrix, b is a full matrix. The operation is done in place.
# the format of a is preserved. 
function hmat_full_add!(a::Hmat, b::AbstractArray{Float64}, scalar, eps=1e-10)
    # println(a)
    # p = to_fmat(a) + scalar*b
    if a.is_fullmatrix
        # a.C += scalar * b
        BLAS.axpy!(scalar, b, a.C)
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_rkmatrix
        C = a.A*a.B'+scalar*b
        # C = copy(b)
        # BLAS.gemm('N','T', 1.0, a.A, a.B, scalar, C);
        a.A, a.B = compress(C, 1e-10) # cautious: rank might increase
        # println(maximum(p-to_fmat(a)))
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_hmat
        m = a.children[1,1].m
        n = a.children[1,1].n
        @views begin
            hmat_full_add!(a.children[1,1], b[1:m,1:n],scalar, eps)
            hmat_full_add!(a.children[1,2], b[1:m,n+1:end],scalar, eps)
            hmat_full_add!(a.children[2,1], b[m+1:end,1:n],scalar, eps)
            hmat_full_add!(a.children[2,2], b[m+1:end,n+1:end],scalar, eps)
        end
        # @assert maximum(p-to_fmat(a))<1e-6
    else
        # println(a)
        error("Should not be here")
    end
end

function full_hmat_add(A::Array{Float64}, H::Hmat)
    if H.is_fullmatrix
        return A + H.C
    end
    if H.is_rkmatrix
        return A + H.A*H.B'
    end
    m = H.m
    n = H.m
    return [
        full_hmat_add(A[1:m,1:n], H.children[1,1]) full_hmat_add(A[1:m, n+1:end], H.children[1,2])
        full_hmat_add(A[m+1:end,1:n], H.children[2,1]) full_hmat_add(A[m+1:end,n+1:end], H.children[2,2])
    ]
end

function compress_low_rank(A::Array{Float64}, B::Array{Float64}, eps::Float64)
    FA = qr(A)
    FB = qr(B)
    W = FA.R*FB.R'
    A1, A2 = compress(W, eps, "svd")
    # @show size(QA), size(A1), size(QB), size(A2)
    return Array(FA.Q)*A1, Array(FB.Q)*A2
end

function rkmat_hmat_add(A::Array{Float64}, B::Array{Float64}, scalar::Float64, H::Hmat, eps::Float64)
    # @show size(A,1), size(A,2),size(B,1), size(B,2),H.m, H.n
    # @assert size(A,2)==size(B,2) && size(A,1)==H.m && size(B,1)==H.n
    eps = 1e-10 #TODO: Change in the future 
    # C = A*B'+scalar*to_fmat(H)
    if H.is_fullmatrix
        C = A*B' + scalar*H.C
        A, B = compress(C, eps)
        # @show "H.is_fullmatrix = ", pointwise_error(A*B', C)
        return A, B
    end

    if H.is_rkmatrix
        A, B = _rkmat_add!(A, B, scalar*H.A, H.B, eps)
        # @show "H.is_rkmatrix = ",pointwise_error(A*B', C)
        return A, B
    end

    if H.is_hmat
        m = H.children[1,1].m
        n = H.children[1,1].n
        A11, B11 = rkmat_hmat_add(A[1:m,:], B[1:n,:],scalar, H.children[1,1], eps)
        A12, B12 = rkmat_hmat_add(A[1:m,:], B[n+1:end,:],scalar, H.children[1,2], eps)
        A21, B21 = rkmat_hmat_add(A[m+1:end,:], B[1:n,:],scalar, H.children[2,1], eps)
        A22, B22 = rkmat_hmat_add(A[m+1:end,:], B[n+1:end,:],scalar, H.children[2,2], eps)
        A0 = [A11 A12 zeros(size(A11,1), size(A21,2)) zeros(size(A11,1), size(A22,2))
              zeros(size(A21,1), size(A11,2)) zeros(size(A21,1), size(A12,2)) A21 A22]
        B0 = [B11   zeros(size(B11,1), size(B12,2)) B21     zeros(size(B11,1), size(B22,2))
            zeros(size(B12,1), size(B11,2)) B12     zeros(size(B12,1), size(B21,2))  B22]
        # @show size(A11), size(B11)
        # @show size(A12), size(B12)
        # @show size(A21), size(B21)
        # @show size(A22), size(B22)
        # println(size(A0), size(B0))
        A, B = compress_low_rank(A0, B0, eps)
        # @show "H.is_hmat = ", pointwise_error(A*B', C)
        return A, B
    end

    error("Invalid")
end

# Perform a = a + scalar * b
# the format of a is preserved
function hmat_add!( a, b, scalar = 1.0, eps=1e-10)
    # p = to_fmat(a) + scalar*to_fmat(b)
    if b.is_fullmatrix
        # println(a)
        @timeit tos "hmat_full_add!" hmat_full_add!(a, b.C, scalar, eps)
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_fullmatrix && b.is_rkmatrix
        if prod(size(b.A))==0 
            return
        end
        # a.C += scalar * b.A * b.B'
        BLAS.gemm!('N','T',scalar, b.A, b.B, 1.0, a.C)
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_fullmatrix && b.is_hmat
        # @timeit tos "full hmat - hadd" begin
        # c = copy(b)
        # @timeit tos "fmat!" to_fmat!(c)
        @timeit tos "fmat!" a.C += scalar * full_hmat_add(a.C, b)
        # a.C += scalar * c.C
    # end
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_rkmatrix && b.is_rkmatrix
        rkmat_add!(a, b, scalar, "svd", eps)
        # println(maximum(p-to_fmat(a)))
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_rkmatrix && b.is_hmat
        # # TODO: is that so?
        # @timeit tos "to_fmat hmat_add!" C = to_fmat(b)    # bottleneck
        # # d = a.A*a.B' + scalar*C
        # BLAS.gemm!('N','T',1.0,a.A, a.B, scalar, C)
        # a.A, a.B = compress(C, eps)
        a.A, a.B = rkmat_hmat_add(a.A, a.B, scalar, b, eps)
        # println(maximum(p-to_fmat(a)))
        # @assert maximum(p-to_fmat(a))<1e-6
        # error("Not implemented yet")
    elseif a.is_hmat && b.is_rkmatrix        
        m = a.children[1,1].m
        n = a.children[1,1].n
        # @views begin
        C11 = rkmat(b.A[1:m,:], b.B[1:n,:], a.s.left, b.t.left)
        C21 = rkmat(b.A[m+1:end,:], b.B[1:n,:], a.s.right, b.t.left)
        C12 = rkmat(b.A[1:m,:], b.B[n+1:end,:], a.s.left, b.t.right)
        C22 = rkmat(b.A[m+1:end,:], b.B[n+1:end,:], a.s.right, b.t.right)
        # end
        hmat_add!(a.children[1,1], C11, scalar, eps)
        hmat_add!(a.children[2,1], C21, scalar, eps)
        hmat_add!(a.children[1,2], C12, scalar, eps)
        hmat_add!(a.children[2,2], C22, scalar, eps)
        # @assert maximum(p-to_fmat(a))<1e-6
    elseif a.is_hmat && b.is_hmat
        for i = 1:2
            for j = 1:2
                hmat_add!(a.children[i,j], b.children[i,j], scalar, eps)
            end
        end
        # @assert maximum(p-to_fmat(a))<1e-6
    end
end

#=
function Base.:+(a::Hmat, b::Hmat)
    # @assert size(a,1) == size(b,1) && size(a,2)==size(b,2)
    c = copy(a)
    hmat_add!( c, b )
    # @assert maximum(abs.(to_fmat(a)+to_fmat(b)-to_fmat(c)))<1e-6
    return c
end
=#

function hadd(a::Hmat, b::Hmat, eps::Float64)
    # @assert size(a,1) == size(b,1) && size(a,2)==size(b,2)
    c = copy(a)
    @timeit tos "hmat_add!" hmat_add!( c, b, 1.0, eps )
    # @assert maximum(abs.(to_fmat(a)+to_fmat(b)-to_fmat(c)))<1e-6
    return c
end

# a -- dense matrix
# b -- hmatrix
function full_mat_mul(a::Hmat, b::Hmat, eps::Float64)
    H = Hmat()
    A = a.C
    if b.is_hmat && (!a.s.isleaf) && (!(a.t.isleaf))
        m, n = b.children[1,1].m, b.children[1,1].n
        p, q = b.m - m, b.n - n
        H.is_hmat = true
        # println(a.is_fullmatrix)
        m0 = a.s.left.N

        H.children = Array{Hmat}([Hmat(m=m0, n=n) Hmat(m=m0, n=q)
                                    Hmat(m=size(a,1)-m0, n=n) Hmat(m=size(a,1)-m0, n=q)])
        a11 = A[1:m0, 1:m]
        a21 = A[m0+1:end, 1:m]
        a12 = A[1:m0, m+1:end]
        a22 = A[m0+1:end, m+1:end]
        b11 = b.children[1,1]
        b12 = b.children[1,2]
        b21 = b.children[2,1]
        b22 = b.children[2,2]

        H.children[1,1] = full_mat_mul(a11, b11, eps) + full_mat_mul(a12, b21, eps)
        H.children[1,2] = full_mat_mul(a11, b12, eps) + full_mat_mul(a12, b22, eps)
        H.children[2,1] = full_mat_mul(a21, b11, eps) + full_mat_mul(a22, b21, eps)
        H.children[2,2] = full_mat_mul(a21, b12, eps) + full_mat_mul(a22, b22, eps)
    elseif b.is_hmat
        H.is_fullmatrix = true
        H.C = A * to_fmat(b)
    elseif b.is_fullmatrix
        H.is_fullmatrix = true
        H.C = A * b.C
    else
        H.is_rkmatrix = true
        H.A = A * b.A
        H.B = b.B
    end
    return H
end

function mat_full_mul(a::Hmat, b::Hmat, eps::Float64)
    B = b.C
    H = Hmat()
    
    if a.is_hmat && (!b.s.isleaf) && (!(b.t.isleaf))
        m, n = a.children[1,1].m, a.children[1,1].n
        p, q = a.m - m, a.n - n
        H.is_hmat = true
        m0 = b.t.left.N

        H.children = Array{Hmat}([Hmat(m=m,n=m0) Hmat(m=m,n=b.n-m0)
                                    Hmat(m=p,n=m0) Hmat(m=p,n=b.n-m0)])
        b11 = B[1:n, 1:m0]
        b12 = B[1:n, m0+1:end]
        b21 = B[n+1:n+q, 1:m0]
        b22 = B[n+1:n+q, m0+1:end]
        a11 = a.children[1,1]
        a12 = a.children[1,2]
        a21 = a.children[2,1]
        a22 = a.children[2,2]

        H.children[1,1] = mat_full_mul(a11, b11, eps) + mat_full_mul(a12, b21, eps) 
        H.children[1,2] = mat_full_mul(a11, b12, eps) + mat_full_mul(a12, b22, eps)
        H.children[2,1] = mat_full_mul(a21, b11, eps) + mat_full_mul(a22, b21, eps)
        H.children[2,2] = mat_full_mul(a21, b12, eps) + mat_full_mul(a22, b22, eps)
    elseif a.is_hmat
        H.is_fullmatrix = true
        H.C = a * B
    elseif a.is_fullmatrix
        H.is_fullmatrix = true
        H.C = a.C * B
    else
        H.is_rkmatrix = true
        H.A = a.A
        H.B = B' * a.B
    end
    return H
end

function transpose_hmat_mat_mul(H::Hmat, V::AbstractArray{Float64})
    if H.is_fullmatrix
        return H.C'*V
    end
    if H.is_rkmatrix
        return H.B * (H.A'*V)
    end
    if H.is_hmat
        h11 = H.children[1,1]
        h12 = H.children[1,2]
        h21 = H.children[2,1]
        h22 = H.children[2,2]
        V1 = transpose_hmat_mat_mul(h11, V[1:h11.m,:]) + transpose_hmat_mat_mul(h21, V[h11.m+1:end,:])
        V2 = transpose_hmat_mat_mul(h12, V[1:h11.m,:]) + transpose_hmat_mat_mul(h22, V[h11.m+1:end,:])
        return [V1;V2]
    end
    error("Invalid H")
end

function hmul(a::Hmat, b::Hmat, eps::Float64)
    # R = to_fmat(a)*to_fmat(b)
    if a.is_fullmatrix
        @timeit tos "full mat" H = full_mat_mul(a, b, eps)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif b.is_fullmatrix
        @timeit tos "mat full" H = mat_full_mul(a, b, eps)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_rkmatrix && b.is_rkmatrix
        @timeit tos "ab" H = Hmat(is_rkmatrix = true, A = a.A, B = b.B * (b.A' * a.B))
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_rkmatrix && b.is_hmat
        # c = copy(b)
        # @timeit tos "to_fmat" to_fmat!(c)
        @timeit tos "BB" BB = transpose_hmat_mat_mul(b, a.B)
        H = Hmat(is_rkmatrix = true, A = a.A, B = BB)
        # H = Hmat(is_rkmatrix = true, A = a.A, B = c.C'*a.B)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_hmat && b.is_rkmatrix
        # c = copy(a)
        # to_fmat!(c)
        @timeit tos "hmat rkmat" H = Hmat(is_rkmatrix = true, A = a*b.A, B = b.B)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_hmat && b.is_hmat 
        H = Hmat(is_hmat = true)
        m = a.children[1,1].m
        n = b.children[1,1].n
        m1 = a.m - m
        n1 = b.n - n
        H.children = Array{Hmat}([Hmat(m=m,n=n) Hmat(m=m,n=n1)
                                 Hmat(m=m1,n=n) Hmat(m=m1,n=n1)])
        for i = 1:2
            for j = 1:2
                # hmat_add!(H.children[i,j], hmul(a.children[i,1],b.children[1,j],eps), 1.0 ,eps)
                # hmat_add!(H.children[i,j], hmul(a.children[i,2],b.children[2,j],eps), 1.0 ,eps)
                H1 = hmul(a.children[i,1],b.children[1,j],eps)
                H2 = hmul(a.children[i,2],b.children[2,j],eps)
                @timeit tos "hadd" H.children[i,j] = hadd( H1, H2, eps)
                # H.children[i,j] = hmul(a.children[i,1],b.children[1,j],eps) + hmul(a.children[i,2],b.children[2,j],eps)
            end
        end
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    else
        error("Undefined")
    end
    H.s = a.s
    H.t = b.t
    H.m = a.m
    H.n = b.n
    # println(maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H))))
    # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    return H
end

function Base.:*(a::Hmat, v::AbstractArray{Float64})
    r = zeros(a.m, size(v,2))
    for i = 1:size(v,2)
        @views hmat_matvec!(r[:,i], a, v[:,i], 1.0)
    end
    return r
end

function Base.:*(v::AbstractArray{Float64}, a::Hmat)
    r = zeros(size(v,1), a.n)
    for i = 1:size(v,1)
        r[i,:] = hmat_matvec2(v[i,:], a, 1.0)
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

# r = r + s*v*a
function hmat_matvec2(v::AbstractArray{Float64}, a::Hmat, s::Float64)
    if a.is_fullmatrix
        res = s*v'*a.C
    elseif a.is_rkmatrix
        res = s*(v'*a.A)*a.B'
    else
        m, n = a.children[1,1].m, a.children[1,1].n
        @views begin
            r1 = hmat_matvec2( v[1:m], a.children[1,1], s) + hmat_matvec2(v[m+1:end], a.children[2,1],  s)
            r2 = hmat_matvec2(v[1:m], a.children[1,2],  s) +  hmat_matvec2( v[m+1:end],a.children[2,2],  s)
        end
        res = [r1;r2]
    end   
    return res[:]
end


function _rkmat_add!(A1, B1, A2, B2, eps)
    if size(A2,2)==0 
        @assert size(B2,2)==0
        return A1, B1
    end

    if size(A1,2)==0
        @assert size(B1,2)==0
        return A2, B2
    end
    
    FAQ, FAR = qr([A1 A2])
    FBQ, FBR = qr([B1 B2])

    W = FAR*FBR'
    U,V = compress(W, eps, "svd")
    r = size(U,1)
    A = FAQ * U
    V = [V;zeros(size(FBQ,1)-size(V,1), size(V,2))]
    B =  FBQ* V # find ways to disable bounds check
    return A, B
end

function rkmat_add!(a, b, scalar, method=1, eps=1e-10)
    A1, B1, A2, B2 = a.A, a.B, scalar*b.A, b.B
    if size(A2,2)==0 
        @assert size(B2,2)==0
        return A1, B1
    end

    if size(A1,2)==0
        @assert size(B1,2)==0
        return A2, B2
    end
    
    FAQ, FAR = qr([A1 A2])
    FBQ, FBR = qr([B1 B2])

    W = FAR*FBR'
    U,V = compress(W, eps, "svd")
    r = size(U,1)
    A = FAQ * U
    V = [V;zeros(size(FBQ,1)-size(V,1), size(V,2))]
    B =  FBQ* V # find ways to disable bounds check
    a.A, a.B= A, B
end



#=
function Base.:*(a::Hmat, b::Hmat)
    # R = to_fmat(a)*to_fmat(b)
    if a.is_fullmatrix
        H = full_mat_mul(a, b)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif b.is_fullmatrix
        H = mat_full_mul(a, b)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_rkmatrix && b.is_rkmatrix
        H = Hmat(is_rkmatrix = true, A = a.A, B = b.B * (b.A' * a.B))
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_rkmatrix && b.is_hmat
        c = copy(b)
        to_fmat!(c)
        H = Hmat(is_rkmatrix = true, A = a.A, B = c.C'*a.B)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_hmat && b.is_rkmatrix
        c = copy(a)
        to_fmat!(c)
        H = Hmat(is_rkmatrix = true, A = c*b.A, B = b.B)
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    elseif a.is_hmat && b.is_hmat 
        H = Hmat(is_hmat = true)
        m = a.children[1,1].m
        n = b.children[1,1].n
        m1 = a.m - m
        n1 = b.n - n
        H.children = Array{Hmat}([Hmat(m=m,n=n) Hmat(m=m,n=n1)
                                 Hmat(m=m1,n=n) Hmat(m=m1,n=n1)])
        for i = 1:2
            for j = 1:2
                H.children[i,j] = a.children[i,1]*b.children[1,j] + a.children[i,2]*b.children[2,j]
            end
        end
        # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    else
        error("Undefined")
    end
    H.s = a.s
    H.t = b.t
    H.m = a.m
    H.n = b.n
    # println(maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H))))
    # @assert maximum(abs.(to_fmat(a)*to_fmat(b)-to_fmat(H)))<1e-6
    return H
end
=#