include("../hmat.jl")

using PyCall
using IterativeSolvers

@pyimport numpy
@pyimport scipy.signal as ss
@pyimport scipy.sparse.linalg as ssl
@pyimport scipy.stats as sstats
@pyimport scipy.special as ssp

function profile(n=10, h=2/2^10, minBlock=64, maxBlock=2^(10-3); printout = true)
    if printout
        println("=======================================================")
        println("== n = $n  minBlock = $minBlock maxBlock = $maxBlock ==")
    end

    xx = collect(1:2^n)*h
    yy = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(1, 5)
    function nf(x,y)
        if x==y
            return -10/h+f(x,y)
        elseif y-x==h
            return f(x,y)
        elseif x-y==h
            return f(x,y)
        else
            return f(x,y)
        end
    end
    t1 = @timed hA = construct1D_low_rank(nf, alpha, beta, h, 1, 2^n, minBlock, maxBlock)
    t6 = @timed A = full_mat(nf, xx, yy)

    y = rand(2^n)
    t7 = @timed A*y
    t8 = @timed F = lu!(A)
    t9 = @timed g = F\y

    

    t2 = @timed begin
        for i = 1:10
            w = hA*y;
        end
    end

    info1 = info(hA)
    t3 = @timed lu!(hA)
    info2 = info(hA)

    w = hA\y;
    t4 = @timed begin
        for i = 1:10
            w = hA\y;
        end
    end

    
    err = norm(w-g)/norm(g)

    if printout
        @printf("Matrix       : full=%d, rk=%d, level=%d\n", info1[1], info1[2], info1[3])
        @printf("Construction : (Hmat)%0.6f sec (Full)%0.6f sec\n", t1[2], t6[2])
        @printf("MatVec       : (Hmat)%0.6f sec (Full)%0.6f sec\n", t2[2]/10, t7[2])
        @printf("LU           : (Hmat)%0.6f sec (Full)%0.6f sec\n", t3[2], t8[2])
        @printf("Solve        : (Hmat)%0.6f sec (Full)%0.6f sec\n", t4[2]/10, t9[2])
        @printf("Error        : %g\n", err)
    end
end

function profile_hmat_only(n=10, h=2/2^10, minBlock=64, maxBlock=2^(10-3); printout = true)
    if printout
        println("=======================================================")
        println("== n = $n  minBlock = $minBlock maxBlock = $maxBlock ==")
    end
    
    x = collect(1:2^n)*h
    y = collect(1:2^n)*h
    f, alpha, beta = Merton_Kernel(1, 5)
    function nf(x,y)
        if x==y
            return -10/h+f(x,y)
        elseif y-x==h
            return f(x,y)
        elseif x-y==h
            return f(x,y)
        else
            return f(x,y)
        end
    end
    t1 = @timed hA = construct1D_low_rank(nf, alpha, beta, h, 1, 2^n, minBlock, maxBlock)
    # t1 = @timed hA = construct1D(test_kerfun, -2^(n-1), 2^(n-1)-1, minBlock, 5, maxBlock)
    y = rand(2^n)

    t4 = @timed begin
        for i = 1:10
            w = hA*y;
        end
    end

    info1 = info(hA)
    # matshow(hA)
    t3 = @timed lu!(hA)
    info2 = info(hA)
    # matshow(hA)

    w = hA\y;
    t5 = @timed begin
        for i = 1:10
            w = hA\y;
        end
    end

    if printout
        @printf("Matrix       : full=%d, rk=%d, level=%d, compress=%0.6f, storage=%g\n", info1[1], info1[2], info1[3],info1[4],(round(info1[4]*hA.m*hA.n)))
        @printf("LU           : full=%d, rk=%d, level=%d, compress=%0.6f, storage=%g\n", info2[1], info2[2], info2[3],info2[4],(round(info2[4]*hA.m*hA.n)))
        @printf("Construction : (Hmat)%0.6f sec \n", t1[2])
        @printf("MatVec       : (Hmat)%0.6f sec \n", t4[2]/10)
        @printf("LU           : (Hmat)%0.6f sec \n", t3[2])
        @printf("Solve        : (Hmat)%0.6f sec \n", t5[2]/10)
    end
    
    # println(tos)
    # reset_timer!(tos)
end

function batch_profile(minBlock=64, offset = 2)
    profile(5, 1/2^5; printout = false)
    for n = [10,11,12,13,14,15,16,17]
        maxBlock = 2^(n-offset)
        profile(n, 1/2^n, minBlock, maxBlock)
    end
end

function batch_profile_hmat_only(minBlock=64, offset = 2)
    profile_hmat_only(5, 1/2^5; printout = false)
    for n = [10,11,12,13,14,15,16,17]
        maxBlock = 2^(n-offset)
        profile_hmat_only(n, 1/2^n, minBlock, maxBlock)
    end
end
