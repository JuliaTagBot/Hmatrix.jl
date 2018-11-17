# solver.jl includes plug-in solvers based on the fast nonlocal operator solver based on HLU preconditioner
include("hmat.jl")
include("hconstruct.jl")


function theta_scheme1D(a=1, b=1, c=-1.0, u0=x->exp(-50x^2), NT=100, T=1, n=5, L=1, minBlock=64, offset=2;printout=true)
    @assert a>=0
    @assert c<=0
    N = 2^n
    h = 2L/(N-1)
    dt = T/NT
    xi = -10*L:h:10*L
    f, alpha, beta = Merton_Kernel(1.0, 5)
    nu = x->f(0,x)
    lambda = sum(nu.(xi))*h
    
    function Afunc_left(x, y)
        if x==y
            return 1+dt/2*(2a/h^2-c+lambda * h)
        elseif abs(y-(x-h))<h/4
            return dt/2*(-a/h^2+b/2h-nu(-h)*h)
        elseif abs(y-(x+h))<h/4
            return dt/2*(-a/h^2-b/2h-nu(h)*h)
        else
            return dt/2*(-nu(y-x)*h)
        end
    end

    function Afunc_right(x, y)
        if x==y
            return 1-dt/2*(2a/h^2-c+lambda * h)
        elseif abs(y-(x-h))<h/4
            return -dt/2*(-a/h^2+b/2h-nu(-h)*h)
        elseif abs(y-(x+h))<h/4
            return -dt/2*(-a/h^2-b/2h-nu(h)*h)
        else
            return -dt/2*(-nu(y-x)*h)
        end
    end

    alpha1 = deepcopy(alpha)
    beta1 = deepcopy(beta)
    for i = 1:length(alpha)
        alpha1[i] = x->-dt/2*h*alpha[i](x)
    end

    alpha2 = deepcopy(alpha)
    beta2 = deepcopy(beta)
    for i = 1:length(alpha)
        alpha2[i] = x->dt/2*h*alpha[i](x)
    end

    t1 = @timed hA = construct1D_low_rank(Afunc_left, alpha1, beta1, h, -2^(n-1), 2^(n-1)-1, minBlock, 2^(n-offset))
    t2 = @timed hB = construct1D_low_rank(Afunc_right, alpha2, beta2, h, -2^(n-1), 2^(n-1)-1, minBlock, 2^(n-offset))

    # A = full_mat(Afunc_left, (-2^(n-1): 2^(n-1)-1)*h, (-2^(n-1): 2^(n-1)-1)*h)
    # B = full_mat(Afunc_right, (-2^(n-1): 2^(n-1)-1)*h, (-2^(n-1): 2^(n-1)-1)*h)
    # check_if_equal(hA, A)
    # check_if_equal(hB, B)
    
    # println(hA.C)
    # return
    # println(cond(hA.C))
    # return
    t3 = @timed lu!(hA)

    U = zeros(2^n, NT+1)
    x = LinRange(-L,L, 2^n)
    U[:,1] = u0.(x)
    
    t4 = @timed begin
        for i = 1:NT
            U[:,i+1] = hA\(hB*U[:,i])
        end
    end
    if printout
        println("""=========== H MATRIX =============
        n=$n
        Explicit Matrix: $(t1[2]) seconds, $(t1[3]) bytes
        Implicit Matrix: $(t2[2]) seconds, $(t2[3]) bytes
        LU: $(t3[2]) seconds, $(t3[3]) bytes
        Iteration: $(t4[2]) seconds, $(t4[3]) bytes""")
    end
    return U
end



function theta_scheme1D_full(a=1, b=1, c=-1.0, u0=x->exp(-50x^2), NT=100, T=1, n=5, L=1; printout=true)
    @assert a>=0
    @assert c<=0
    N = 2^n
    h = 2L/(N-1)
    dt = T/NT
    xi = -10*L:h:10*L
    f, alpha, beta = Merton_Kernel(1.0, 10)
    nu = x->f(0,x)
    lambda = sum(nu.(xi))*h
    
    function Afunc_left(x, y)
        if x==y
            return 1+dt/2*(2a/h^2-c+lambda * h)
        elseif abs(y-(x-h))<h/4
            return dt/2*(-a/h^2+b/2h-nu(-h)*h)
        elseif abs(y-(x+h))<h/4
            return dt/2*(-a/h^2-b/2h-nu(h)*h)
        else
            return dt/2*(-nu(y-x)*h)
        end
    end

    function Afunc_right(x, y)
        if x==y
            return 1-dt/2*(2a/h^2-c+lambda * h)
        elseif abs(y-(x-h))<h/4
            return -dt/2*(-a/h^2+b/2h-nu(-h)*h)
        elseif abs(y-(x+h))<h/4
            return -dt/2*(-a/h^2-b/2h-nu(h)*h)
        else
            return -dt/2*(-nu(y-x)*h)
        end
    end


    t1 = @timed A = full_mat(Afunc_left, (-2^(n-1): 2^(n-1)-1)*h, (-2^(n-1): 2^(n-1)-1)*h)
    t2 = @timed B = full_mat(Afunc_right, (-2^(n-1): 2^(n-1)-1)*h, (-2^(n-1): 2^(n-1)-1)*h)
    
    t3 = @timed F = lu!(A)

    U = zeros(2^n, NT+1)
    x = LinRange(-L,L, 2^n)
    U[:,1] = u0.(x)
    
    t4 = @timed begin
        for i = 1:NT
            U[:,i+1] = F\(B*U[:,i])
        end
    end
    if printout
        println("""=========== FULL MATRIX =============
        n=$n
        Explicit Matrix: $(t1[2]) seconds, $(t1[3]) bytes
        Implicit Matrix: $(t2[2]) seconds, $(t2[3]) bytes
        LU: $(t3[2]) seconds, $(t3[3]) bytes
        Iteration: $(t4[2]) seconds, $(t4[3]) bytes""")
    end
    return U
end


function batch1(minBlock=64, offset=3)
    a = 1.0
    b = 1.0
    c = -1.0
    u0 = x->exp(-50x^2)
    L = 1.0
    NT = 1000
    T = 1.0
    theta_scheme1D(a, b, c, u0, NT, T, 5, L,  minBlock, offset; printout=false)
    for n = [10, 11, 12, 13, 14, 15, 16]
        theta_scheme1D(a, b, c, u0, NT, T, n, L,  minBlock, offset)
    end
end

function batch2(minBlock=64, offset=3)
    a = 1.0
    b = 1.0
    c = -1.0
    u0 = x->exp(-50x^2)
    L = 1.0
    NT = 1000
    T = 1.0
    theta_scheme1D_full(a, b, c, u0, NT, T, 5, L; printout=false)
    for n = [10, 11, 12, 13, 14, 15, 16]
        theta_scheme1D_full(a, b, c, u0, NT, T, n, L)
    end
end