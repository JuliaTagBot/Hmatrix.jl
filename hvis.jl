include("hmat.jl")
include("hexample.jl")

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



