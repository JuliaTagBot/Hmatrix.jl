using LinearAlgebra
include("hmat.jl")

function generate_hmat()
    N = 3070
    cs = uniform_cluster(N, 57)
    c = cluster_from_list(cs)
    A = zeros(N, N)
    for i = 1:size(A,1)
        for j = 1:size(A,1)
            if i==j
                A[i,j] = 10
            else
                A[i,j] = -1/(abs(j-i))
            end
        end
    end
    Nleaf = 64
    eps = 1e-5
    Rrank = 10
    MaxBlock = div(size(A,1),2)
    
    H,_ = construct_hmat(A, c, Nleaf, Rrank, eps, MaxBlock)
    verify_matvec_error(H, A)
    verify_matrix_error(H,A)
    verify_lu_error(H)
    
    return A, H
end

