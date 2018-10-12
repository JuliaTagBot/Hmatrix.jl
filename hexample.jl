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
        A[i,i] = -sum(A[1,:])*2
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