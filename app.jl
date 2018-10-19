include("hmat.jl")
include("hexample.jl")
include("hconstruct.jl")

using PyCall
using IterativeSolvers

@pyimport numpy
@pyimport scipy.signal as ss
@pyimport scipy.sparse.linalg as ssl
@pyimport scipy.stats as sstats
@pyimport scipy.special as ssp


function profile(n=10, h=2/2^10)
    println("=======================================================")
    println("======================= n = $n ========================")
    
    x = collect(1:2^n)*h
    y = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(1, 10)
    function nf(x,y)
        if x==y
            return -10
        else
            return f(x,y)
        end
    end
    t1 = @timed hA = construct1D_low_rank(nf, alpha, beta, h, 1, 2^n, 32, 2^(n-3))
    t2 = @timed A = full_mat(nf, x, y)
    y = rand(2^n)

    info1 = info(hA)
    t3 = @timed lu!(hA)
    info2 = info(hA)

    w = hA\y;
    t4 = @timed begin
        for i = 1:10
            w = hA\y;
        end
    end

    F = lu(A)
    t5 = @timed F\y
    g = A\y;
    err = norm(w-g)/norm(g)
    t6 = @timed lu(A)


    @printf("Matrix       : full=%d, rk=%d, level=%d\n", info1[1], info1[2], info1[3])
    @printf("Construction : (Hmat)%0.6f sec (Full)%0.6f sec\n", t1[2], t2[2])
    @printf("LU           : (Hmat)%0.6f sec (Full)%0.6f sec\n", t3[2], t6[2])
    @printf("MatVec       : (Hmat)%0.6f sec (Full)%0.6f sec\n", t4[2]/10, t5[2])
    @printf("Error        : %g\n", err)

end

function profile_hmat_only(n=10, h=2/2^10, minBlock=64, maxBlock=2^(10-4))
    println("=======================================================")
    println("======================= n = $n ========================")
    
    x = collect(1:2^n)*h
    y = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(1, 10)
    function nf(x,y)
        if x==y
            return -2/h+f(x,y)
        elseif y-x==h
            return 1/h+f(x,y)
        elseif x-y==h
            return 1/h+f(x,y)
        else
            return f(x,y)
        end
    end
    t1 = @timed hA = construct1D_low_rank(nf, alpha, beta, h, 1, 2^n, minBlock, maxBlock)
    y = rand(2^n)

    info1 = info(hA)
    t3 = @timed lu!(hA)
    info2 = info(hA)

    w = hA\y;
    t4 = @timed begin
        for i = 1:10
            w = hA\y;
        end
    end

    @printf("Matrix       : full=%d, rk=%d, level=%d, compress=%0.6f, storage=%g\n", info1[1], info1[2], info1[3],info1[4],(round(info1[4]*hA.m*hA.n)))
    @printf("Construction : (Hmat)%0.6f sec \n", t1[2])
    @printf("LU           : (Hmat)%0.6f sec \n", t3[2])
    @printf("MatVec       : (Hmat)%0.6f sec \n", t4[2]/10)

end

function batch_profile()
    profile(5, 1/2^5)
    for n = [9,10,11,12,13,14]
        profile(n, 1/2^n)
    end
end

function batch_profile_hmat_only(minBlock=64)
    profile_hmat_only(5, 1/2^5)
    for n = [9,10,11,12,13,14,15,16,17]
        maxBlock = 2^(n-3)
        profile_hmat_only(n, 1/2^n, minBlock, maxBlock)
    end
end
