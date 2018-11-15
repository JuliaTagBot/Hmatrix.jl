include("hmat.jl")
# include("hconstruct.jl")
# include("hexample.jl")

using PyCall
using IterativeSolvers

@pyimport numpy
@pyimport scipy.signal as ss
@pyimport scipy.sparse.linalg as ssl
@pyimport scipy.stats as sstats
@pyimport scipy.special as ssp


function pygmres(A, x, op=nothing)
    N = length(x)
    # convert A
    if typeof(A)==Array{Float64,2}|| typeof(A)==Hmat
        lo = ssl.LinearOperator((N, N), matvec=x->A*x)
    else
        lo = ssl.LinearOperator((N, N), matvec=A)
    end

    # make LinearOperator
    if op==nothing
        Mop = ssl.LinearOperator((N, N), matvec=x->x)
    else
        Mop = ssl.LinearOperator((N, N), matvec=op)
    end

    # solve
    y = ssl.gmres(lo, x, M = Mop)[1]
    return y
end

function pycallback(rk)
    global cnt
    global err
    println("Iteration $cnt, Error = $(err[end])")
    push!(err, norm(rk))
    cnt += 1
end

function pygmres_with_call_back(A, x, op=nothing)
    global err
    global cnt
    cnt = 0
    err = []
    N = length(x)
    # convert A
    if typeof(A)==Array{Float64,2} || typeof(A)==Hmat
        lo = ssl.LinearOperator((N, N), matvec=x->A*x)
    else
        lo = ssl.LinearOperator((N, N), matvec=A)
    end

    # make LinearOperator
    if op==nothing
        Mop = ssl.LinearOperator((N, N), matvec=x->x)
    else
        Mop = ssl.LinearOperator((N, N), matvec=op)
    end
    # solve
    y = ssl.gmres(lo, x, callback = PyCall.jlfun2pyfun(pycallback), M = Mop)[1]
    return y, Array{Float64}(err)
end

#=
function iterative_hmat(n=10, minN=16, eps=1e-6, maxR=8, maxN = 256)
    println("=======================================================")
    println("n=$(2^n), minN=$minN, maxN=$maxN, eps=$eps, maxR=$maxR")
    # s = 0.8
    # A = fraclap(n, 0.8)
    if false
        n = 2^n
        nn = Int(n/2)
        hA = construct1D(test_kerfun, -nn, nn-1, minN, maxR, maxN)
        A = zeros(n, n)
        for i = -Int(n/2):Int(n/2)-1
            for j = -Int(n/2):Int(n/2)-1
                A[i+Int(n/2)+1, j+Int(n/2)+1] = test_kerfun(i, j)
            end
        end
    else
        A = realmatrix(2^n)
        @time hA = construct_hmat(A, minN, eps, maxR, maxN);
    end
    y = rand(size(A,1))
    g = A\y
    # w = pygmres(x->A*x, y)
    
    @time lu!(hA);
    cnt = 0
    w, cnt = pygmres_mat(x->A*x, y)
    @printf("no preconditioner count=\033[32;1;4m%d\033[0m, Error=%0.6g \n", cnt, norm(g-w)/norm(g))
    w, cnt = pygmres_mat(x->A*x, y, x->hA\x)
    @printf("H-mat count            =\033[32;1;4m%d\033[0m, Error=%0.6g \n", cnt, norm(g-w)/norm(g))

end

function batch_iterative_hmat()
    # speed up around 10 times.
    # Factorization around 10 times.
    for n = 0:5
        M = 2^n
        iterative_hmat(9+n, 16*M, 1e-5, 8,  512)
    end
end
=#