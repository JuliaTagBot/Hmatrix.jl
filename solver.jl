# solver.jl includes plug-in solvers based on the fast nonlocal operator solver based on HLU preconditioner
include("hmat.jl")

# [a,b] computational domain, 2^n number of intervals
# [0,T] time evolution region, M intervals
# kfun(i, j) kernel function
# qfun(t) query functions -- return fDJ
# ffun(x, t) -- source term
function theta_scheme1D(a, b, n, T, M, kfun, qfun, ffun, theta)
    kfun2 = (i,j)->1-theta*dt*kfun(i,j)
    nn = 2^(n-1)
    maxN = Int(nn/4)
    maxR = 32
    minN = 64
    t1 = @timed hA = construct1D(kfun2, -nn, nn-1, minN, maxR, maxN)
    t2 = @timed hB = construct1D(kfun, -nn, nn-1, minN, maxR, maxN)
    t3 = @timed lu!(hA)

    U = zeros(2^n, M+1)
    x = LinRange(a, b, n)
    U[:,1] = u0.(x)
    dt = T/M
    t4 = @timed begin
        for i = 1:M
            t = dt*i
            t_ = dt*(i-1)
            t_theta = dt*i - (1-theta)*dt
            F = dt*ffun.(x, t_theta) + dt * theta * qfun.(x, t) + dt * (1-theta)*qfun.(x,t_) 
            U[:,i+1] = u0.(x)\(F + dt*(1-theta)*hB*U[:,i-1]+U[:,i-1])
        end
    end
    info = """Explicit Matrix: $(t1[2]) seconds, $(t1[3]) bytes
Implicit Matrix: $(t2[2]) seconds, $(t2[3]) bytes
LU: $(t3[2]) seconds, $(t3[3]) bytes
Iteration: $(t4[2]) seconds, $(t4[3]) bytes"""
    return U, info 
end



    