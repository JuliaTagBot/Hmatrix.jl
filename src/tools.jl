
export 
gmres,
bicgstab

const LinearOperator = Union{Nothing, Array{Float64,2}, Hmat, Function}
function gmres(A::LinearOperator, b::Array{Float64}, x0=nothing; 
        tol::Float64=1e-05, restart::Union{Int64, Nothing}=nothing, maxiter::Union{Int64, Nothing}=nothing, 
        M::LinearOperator=nothing, callback::Union{Function, Nothing}=nothing)
    N = length(b)
    # convert A
    if typeof(A)==Array{Float64,2} || typeof(A)==Hmat
        lo = ssl.LinearOperator((N, N), matvec=x->A*x)
    elseif isa(A, Function)
        lo = ssl.LinearOperator((N, N), matvec=A) # A is a linear operator
    end

    if M!=nothing        
        if typeof(M)==Array{Float64,2} || typeof(M)==Hmat
            Mop = ssl.LinearOperator((N, N), matvec=x->M\x)
        else
            Mop = ssl.LinearOperator((N, N), matvec=M)
        end
    else
        Mop = nothing
    end
    # solve
    y, info = ssl.gmres(lo, b, x0, tol=tol, restart = restart, maxiter=maxiter, M=Mop, callback=callback)
    if info<0
        @warn "illegal input or breakdown"
    elseif info>0
        @warn "convergence to tolerance not achieved, number of iterations"
    end
    return y
end

function bicgstab(A::LinearOperator, b::Array{Float64}, x0=nothing; 
    tol::Float64=1e-05, maxiter::Union{Int64, Nothing}=nothing, 
    M::LinearOperator=nothing, callback::Union{Function, Nothing}=nothing)
    N = length(b)
    # convert A
    if typeof(A)==Array{Float64,2} || typeof(A)==Hmat
        lo = ssl.LinearOperator((N, N), matvec=x->A*x)
    elseif isa(A, Function)
        lo = ssl.LinearOperator((N, N), matvec=A) # A is a linear operator
    end

    if M!=nothing        
        if typeof(M)==Array{Float64,2} || typeof(M)==Hmat
            Mop = ssl.LinearOperator((N, N), matvec=x->M\x)
        else
            Mop = ssl.LinearOperator((N, N), matvec=M)
        end
    else
        Mop = nothing
    end
    # solve
    y, info = ssl.bicgstab(lo, b, x0, tol=tol, maxiter=maxiter, M=Mop, callback=callback)
    if info<0
        @warn "illegal input or breakdown"
    elseif info>0
        @warn "convergence to tolerance not achieved, number of iterations"
    end
    return y
end