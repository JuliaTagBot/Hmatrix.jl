include("third-party/fast.jl")
function fraclap(n, s)
    N = 2^n
    A = zeros(N,N)
    for i = 1:N
        for j = 1:N
            if i==j
                continue
            else
                A[i,j] = 1/abs(i-j)^(2s)
            end
        end
    end
    for i = 1:N
        A[i,i] = -sum(A[1,:])*2 - 1.5
    end
    return A
end

function fraclap2(N, s)
    A = zeros(N,N)
    for i = 1:N
        for j = 1:N
            if i==j
                continue
            else
                A[i,j] = 1/abs(i-j)^(2s)
            end
        end
    end
    for i = 1:N
        A[i,i] = -sum(A[1,:])*2 - 1.5
    end
    return A
end

function fraclap_noise(n, s)
    N = 2^n
    A = zeros(N,N)
    for i = 1:N
        for j = 1:N
            if i==j
                continue
            else
                A[i,j] = 1/abs(i-j)^(2s) * (1+0.01*rand())
            end
        end
    end
    for i = 1:N
        A[i,i] = -sum(A[1,:])*2
    end
    return A
end

function realmatrix(N)
    m = Int((N-2)/2) # resolution: [a,b] is split into 2m+1 intervals
    s = 0.8; alpha=2s
    Lw = 2.0 # near-field range
    n = x->1/abs(x)^(1+2s) # kernel
    r = 0.3 # window function
    a = -1.0
    b = 1.0
    N = 2m # must be consistent N = m*Int(Lw/L)
    gW = 1/(s*Lw^(2s)) # far-field contribution
    fWx = x->0 # int_{|x+y|>=L_W} u(x+y)n(y) dy
    u = zeros(2(N+m)+1) # must be consistent

    c0 = C_n_s(1, s)
    c1 = 2^(-alpha)*gamma(1/2)/gamma(1+alpha/2)/gamma((1+alpha)/2)

    M = compute_M(Lw, n, r, N, gW)
    K = compute_K(M, m)
    fW = compute_fW(-1., 1., m, fWx)
    hW = compute_hW(M, u, N, m)
    # @show M
    
    K *= -c0
    return K[1:end-1,1:end-1]
end