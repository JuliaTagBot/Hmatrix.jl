include("hmat.jl")
include("hexample.jl")

function showmat(A)
    println("===================")
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            @printf("%0.3f ", A[i,j])
        end
        println("")
    end
    println("===================")
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


function test_fraclap()
    n = 10
    s = 0.5
    C = fraclap(n, s)
    H = construct_hmat(C, 16, 1e-6, 8)
    plot_hmat(H)
end


function test_construct_hmat()
    n = 10
    s = 0.5
    C = fraclap(n, s)
    for eps = [1e-1,1e-4,1e-8,1e-12]
        H = construct_hmat(C, 16, eps, 8)
        to_fmat!(H)
        @show eps, maximum(abs.(C-H.C))
    end
end

function test_hmat_add()
    n = 10
    C = fraclap(n, 0.5)
    D = fraclap(n, 0.8)
    for eps = [1e-1,1e-4,1e-8, 1e-10]
        H1 = construct_hmat(C, 16, eps, 8)
        H2 = construct_hmat(D, 16, eps, 8)
        hmat_add!(H1, H2, 2.0)
        to_fmat!(H1)
        @show maximum(abs.(H1.C-C-2*D))
    end

    for eps = [1e-1,1e-4,1e-8, 1e-10]
        H1 = construct_hmat(C, 16, eps, 8)
        H2 = construct_hmat(D, 32, eps, 8)
        hmat_add!(H1, H2, 2.0)
        to_fmat!(H1)
        @show maximum(abs.(H1.C-C-2*D))
    end
end

function test_hmat_add2()
    n = 10
    C = fraclap(n, 0.5)
    D = fraclap(n, 0.8)
    for eps = [1e-1,1e-4,1e-8, 1e-10]
        H1 = construct_hmat(C, 16, eps, 8)
        H2 = construct_hmat(D, 32, eps, 8)
        H = H1 + H2
        to_fmat!(H)
        @show maximum(abs.(H.C-C-D))
    end
end

function test_hmat_multiply()
    n = 10
    C = fraclap(n, 0.5)
    D = fraclap(n, 0.8)
    for eps = [1e-1,1e-4,1e-8, 1e-10]
        H1 = construct_hmat(C, 16, eps, 8)
        H2 = construct_hmat(D, 16, eps, 8)
        H = H1*H2
        to_fmat!(H)
        @show maximum(abs.(H.C-C*D))
    end
end


function test_hmat_transpose()
    n = 10
    C = fraclap_noise(n, 0.5)
    eps = 1e-6
    H1 = construct_hmat(C, 16, eps, 8)
    transpose!(H1)
    to_fmat!(H1)
    @show maximum(abs.(H1.C-C'))
    # matshow(H1.C-C')
    # println(C')
    # for eps = [1e-1,1e-4,1e-8, 1e-10]
    #     H1 = construct_hmat(C, 16, eps, 8)
    #     H2 = construct_hmat(D, 16, eps, 8)
    #     H = H1*H2
    #     to_fmat!(H)
    #     @show maximum(abs.(H.C-C*D))
    # end
end

function test_hmat_lu()
    n = 10
    C = fraclap(n, 0.5)
    eps = 1e-6
    H1 = construct_hmat(C, 16, eps, 8)
    # plot_hmat(H1)
    hmat_lu!(H1)
end


function test_hmat_trisolve()
    A = rand(10,10)
    A = Array{Float64}(LowerTriangular(A))
    B = rand(10,10)
    hA = fmat(A)
    hB = fmat(B)
    hmat_trisolve!(hA, hB, true, false, false)
    println(norm(hB.C-A\B, Inf))

    A = rand(10,10)
    A = Array{Float64}(LowerTriangular(A))
    B = rand(10,10)
    hA = fmat(A)
    hB = fmat(B)
    hA.P = [10,8,9,7,6,5,4,3,2,1]
    hmat_trisolve!(hA, hB, true, false, true)
    println(norm(hB.C-A\B[hA.P,:], Inf))

    A = rand(10,10)
    A = Array{Float64}(LowerTriangular(A))+UniformScaling(1.0)-diagm(0=>diag(A))
    B = rand(10,1)
    hA = fmat(A)
    hB = rkmat(B, B)
    C = B*B'
    hA.P = [10,8,9,7,6,5,4,3,2,1]
    hmat_trisolve!(hA, hB, true, true, true)
    to_fmat!(hB)
    println(norm(hB.C-A\C[hA.P,:], Inf))

    A = rand(10,10)
    A = Array{Float64}(UpperTriangular(A))
    B = rand(10,10)
    hA = fmat(A)
    hB = fmat(B)
    hmat_trisolve!(hA, hB, false, false, false)
    println(norm(hB.C-B/A, Inf))


    n = 10
    s = 0.5
    A = fraclap(n, 0.8)
    B = fraclap(n, 0.2)
    A = Array{Float64}(LowerTriangular(A))
    
    # @show size(B)
    hA = construct_hmat(A, 16, 1e-8, 8)
    hB = construct_hmat(B, 16, 1e-8, 8)
    # plot_hmat(hA)
    # @show size(B)
    hmat_trisolve!(hA, hB, true, false, false)
    to_fmat!(hB)
    # @show size(B)
    println(norm(hB.C-A\B, Inf))

end

function test_lu()
    # A = rand(10,10)
    # hA = fmat(A)
    # hmat_lu!(hA)
    # to_fmat!(hA)

    for eps = [1e-3, 1e-6, 1e-8, 1e-10]
        n = 10
        s = 0.5
        A = fraclap(n, 0.8)
        hA = construct_hmat(A, 16, eps, 8)
        hmat_lu!(hA)
        to_fmat!(hA)
        
        U = UpperTriangular(hA.C)
        L = LowerTriangular(hA.C)-diagm(0=>diag(hA.C))+UniformScaling(1.0)
        Q = copy(A)
        Q = Q[hA.P,:]
        println(norm(L*U-Q, Inf))
    end
end

function test_matvec()
    eps = 1e-2
    n = 10
    s = 0.5
    A = fraclap(n, 0.2)
    hA = construct_hmat(A, 16, eps, 8)
    y = rand(2^n)
    y1 = hA*y
    y2 = A*y
    println(norm(y1-y2)/norm(y2))
end

function test_solve()
    # A = rand(100,100)
    # hA = fmat(A)
    # hmat_lu!(hA)
    # to_fmat!(hA)

    eps = 1e-6
    n = 5
    s = 0.5
    A = fraclap(n, 0.8)
    hA = construct_hmat(A, 16, eps, 8)
    hmat_lu!(hA)

    hB = Hmat()
    hmat_copy!(hB, hA)
    to_fmat!(hB)
    U = UpperTriangular(hB.C)
    L = LowerTriangular(hB.C)-diagm(0=>diag(hB.C))+UniformScaling(1.0)
    Q = copy(A)
    Q = Q[hB.P,:]
    println(norm(L*U-Q, Inf))

    y = rand(hA.m)
    g1 = hmat_solve(hA, y, true)

    w = y[hB.P]
    g2 = L\w

    println(norm(g1-g2)/norm(g2))
    
    s1 =  hmat_solve(hA, g2, false)
    s2 = U\g2
    println(norm(s1-s2)/norm(s2))
        
        
end

function test_solve2()
    for eps = [1e-3, 1e-6, 1e-8, 1e-10]
        n = 5
        s = 0.5
        A = fraclap(n, 0.8)
        hA = construct_hmat(A, 16, eps, 8)
        hmat_lu!(hA)

        y = rand(size(A,1))
        g1 = copy(y)
        w = hA\g1
        w2 = A\y
        println(norm(w-w2)/norm(w2))
    end

    # hB = Hmat()
    # hmat_copy!(hB, hA)
    # to_fmat!(hB)
    # U = UpperTriangular(hB.C)
    # L = LowerTriangular(hB.C)-diagm(0=>diag(hB.C))+UniformScaling(1.0)
    # Q = copy(A)
    # Q = Q[hB.P,:]
    # println(norm(L*U-Q, Inf))

    # showmat(L)
    # showmat(U)
    # println(hB.P)
    
    # y = rand(hA.m)
    # w = y[hB.P]
    # g2 = L\w
    # s2 = U\g2

    # g1 = copy(y)
    # hmat_solve!(hA, g1, true)    
    # println(norm(g1-g2)/norm(g2))
    # hmat_solve!(hA, g1, false)
    # println(norm(g1-s2)/norm(s2))

    # s0 = A\y
    # s3 = U\(L\y[hB.P])
    # println(norm(s0-s3)/norm(s0))
    # println(norm(s0-s2)/norm(s0))

    
end

