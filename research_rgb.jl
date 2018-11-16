using LinearAlgebra
include("hconstruct.jl")

function generate_hmat()
    N = 3019
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
    H = construct_hmat(A, Nleaf, eps, Rrank, MaxBlock)
    verify_lu_error(H)
    verify_matvec_error(H, A)
    verify_matrix_error(H,A)
    return A, H
end

function mm()
    G = [+1.0000 -1.0000 -0.5000 -0.3333 -0.2500 -0.2000 -0.1667 -0.1429
    -1.0000 +1.0000 -1.0000 -0.5000 -0.3333 -0.2500 -0.2000 -0.1667
    -0.5000 -1.0000 +1.0000 -1.0000 -0.5000 -0.3333 -0.2500 -0.2000
    -0.3333 -0.5000 -1.0000 +1.0000 -1.0000 -0.5000 -0.3333 -0.2500
    -0.2500 -0.3333 -0.5000 -1.0000 +1.0000 -1.0000 -0.5000 -0.3333
    -0.2000 -0.2500 -0.3333 -0.5000 -1.0000 +1.0000 -1.0000 -0.5000
    -0.1667 -0.2000 -0.2500 -0.3333 -0.5000 -1.0000 +1.0000 -1.0000
    -0.1429 -0.1667 -0.2000 -0.2500 -0.3333 -0.5000 -1.0000 +1.0000]

    R = [+1.0000 -1.0000 -0.5000 -0.3333 -0.2500 -0.2000 -0.1667 -0.1429
    -0.5000 -1.5000 +0.7500 -1.1667 -0.6250 -0.4333 -0.3333 -0.2714
    -0.3333 +0.5556 -1.5833 +1.5370 -0.7361 -0.3259 -0.2037 -0.1468
    -1.0000 -0.0000 +0.9474 -2.2895 +0.1140 -0.1412 -0.1737 -0.1704
    -0.2500 +0.3889 +0.5789 +0.6637 +1.5310 -0.5991 -0.1788 -0.0654
    -0.2000 +0.3000 +0.4158 +0.3738 -0.3913 +1.0439 -0.8537 -0.3480
    -0.1667 +0.2444 +0.3263 +0.2644 -0.0427 -0.3333 -1.1567 +0.9914
    -0.1429 +0.2063 +0.2692 +0.2056 -0.1168 -0.8178 -0.3865 -0.7735]
    A = G[1:4,1:4]
    B = G[1:4,5:end]
    C = G[5:end, 1:4]
    D = G[5:end, 5:end]

    AA = R[1:4,1:4]
    BB = R[1:4,5:end]
    CC = R[5:end, 1:4]
    DD = R[5:end, 5:end]

    F = lu!(A)
    println(maximum(abs.(A-AA)))
    B = F.L\B[F.p,:]
    println(maximum(abs.(B-BB)))
    C = C/F.U
    println(maximum(abs.(C-CC)))
    D = D-C*B
    DF = lu!(D)
    println(maximum(abs.(D-DD)))
    
    C = C[DF.p,:]   # you need to permutate!
    H = [A B
        C D]
    P = [F.p;DF.p .+ 4]
    println(P)
    M = G[P,:] - (LowerTriangular(H)-diagm(0=>diag(H))+UniformScaling(1.0))*UpperTriangular(H)
    printmat(G[P,:]-(LowerTriangular(H)-diagm(0=>diag(H))+UniformScaling(1.0))*UpperTriangular(H))
    # printmat()
    println(maximum(abs.(M)))

end

function mm2(G, m)
    A = G[1:m,1:m]
    B = G[1:m,m+1:end]
    C = G[m+1:end, 1:m]
    D = G[m+1:end, m+1:end]

    F = lu!(A)
    B = F.L\B[F.p,:]
    println("Permutation: $(F.p)")
    C = C/F.U
    D = D-C*B
    DF = lu!(D)
    
    C = C[DF.p,:]   # you need to permutate!
    H = [A B
        C D]
    P = [F.p;DF.p .+ m]
    return H, P
    # M = G[P,:] - (LowerTriangular(H)-diagm(0=>diag(H))+UniformScaling(1.0))*UpperTriangular(H)
    # printmat(G[P,:]-(LowerTriangular(H)-diagm(0=>diag(H))+UniformScaling(1.0))*UpperTriangular(H))
    # printmat()
    # println(maximum(abs.(M)))
end


# A, H = generate_hmat()
# verify_matrix_error(H,A)
# verify_matvec_error(H,A)
# G = verify_lu_error(H);

function test_lu()
    F = [2.0 1.0
    1.0 2.0]
    LR = [1.0 1.0
    1.0 1.0]
    A = [F F LR LR
        F 2*F F LR
        LR F 2*F F 
        LR LR F 2*F]
    H = construct_hmat(A, 2, 1e-12, 1, 2)
    # println(H)
    lu!(H)
    G = to_fmat(H)
    L = LowerTriangular(G)-diagm(0=>diag(G))+UniformScaling(1.0)
    U = UpperTriangular(G)
    printmat(L*U-A)
    # return H
end

#=
function readfile(filename)
    S = read(filename, String)
    M = map(x->parse(Float64, x), split(S))
    n = Int(round(sqrt(length(M))))
    copy(reshape(M, n, n)')
end

using DelimitedFiles
N = readdlm("flap/A_MQ.txt")
H1 = construct_hmat(N, 80, 1e-2, 30, -1)
H2 = construct_hmat(N, 80, 5e-2, 10, -1)

HH1 = to_fmat(H1)
y1 = norm(HH1-N)/norm(N)
y = rand(H1.m)
y2 = norm(H1*y - N*y)/norm(N*y)
println("Matrix Error = $y1, Matvec Error = $y2")

HH2 = to_fmat(H2)
y1 = norm(HH2-N)/norm(N)
y = rand(H2.m)
y2 = norm(H2*y - N*y)/norm(N*y)
println("Matrix Error = $y1, Matvec Error = $y2")

figure()
matshow(H1)
figure()
matshow(H2)
include("tools.jl")
b = readdlm("flap/b.txt")
x0 = N\b;

g = pygmres_with_call_back(H1, b)
=#