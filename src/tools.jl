# tools.jl
function pygmres(A, x, op=nothing; tol=1e-10, maxiter = 1000)
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
        
        if typeof(op)==Array{Float64,2} || typeof(op)==Hmat
            Mop = ssl.LinearOperator((N, N), matvec=x->op\x)
        else
            Mop = ssl.LinearOperator((N, N), matvec=op)
        end
    end

    # solve
    y = ssl.gmres(lo, x, M = Mop, tol = tol, maxiter=maxiter)[1]
    return y
end

function pycallback(rk)
    global cnt
    global err
    global verbose_
    push!(err, norm(rk))
    if verbose_
        println("Iteration $cnt, Error = $(err[end])")
    end
    cnt += 1
end

function pygmres_with_call_back(A, x, op=nothing, verbose=true)
    global err
    global cnt
    global verbose_
    verbose_ = verbose
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
        
        if typeof(op)==Array{Float64,2} || typeof(op)==Hmat
            Mop = ssl.LinearOperator((N, N), matvec=x->op\x)
        else
            Mop = ssl.LinearOperator((N, N), matvec=op)
        end
    end
    # solve
    y = ssl.gmres(lo, x, callback = PyCall.jlfun2pyfun(pycallback), M = Mop, tol=1e-8, maxiter=1000)[1]
    return y, Array{Float64}(err)
end

function hprecond(Hpred_::Hmat, H::Union{Hmat, Array{Float64,2}}, A::Array{Float64})
    Hpred = copy(Hpred_)
    lu!(Hpred)
    x = rand(size(A,1))
    b = A*x
    y2, err2 = pygmres_with_call_back(H, b, nothing, true)
    y1, err1 = pygmres_with_call_back(H, b, Hpred, true)
    
    println("Error1 = ", rel_error(y1, x))
    println("Error2 = ", rel_error(y2, x))
    semilogy(err1,"o-",label="With Preconditioner")
    semilogy(err2, "+-", label="Without Preconditioner")
    legend()
    xlabel("Iteration")
    ylabel("Error")
    savefig("Iter.png")
    close("all")
end