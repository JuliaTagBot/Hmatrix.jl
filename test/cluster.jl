@testset "NewHmat" begin
    X = LinRange(0,10,2000) |> collect
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 25
    c, H = NewHmat()
    @test maximum(abs.(FullMat(Hparams.Kernel, c.X, c.X)-Array(H)))<1e-8
end

@testset "NewHmat2" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(100+sum((x-y).^2))
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    e = maximum(abs.(FullMat(Hparams.Kernel, c.X, c.X)-Array(H)))
    @info "Error = $e" 
end

function Merton_Kernel(eps, r)
    f = (x,y)->exp(-eps^2*(x-y)^2)
    alpha = Function[]
    beta = Function[]
    for i = 0:r-1
        push!(alpha, t->exp(-eps^2*t^2)*t^i*2^i*eps^(2i)/factorial(i))
        push!(beta, t->exp(-eps^2*t^2)*t^i)
    end
    return f, alpha, beta
end

function mk2d_1(t, s, m, n, eps)
    return (2*eps^2)^(m+n)/factorial(m)/factorial(n)*s^m*t^n*exp(-eps^2*(t^2+s^2))
end

function mk2d_2(t, s, m, n, eps)
    return t^n*s^m*exp(-eps^2*(t^2+s^2))
end

function Merton_Kernel2D(e, r)
    f = (x,y)->exp(-e^2*norm(x-y)^2)
    alpha = Function[]
    beta = Function[]
    for m = 0:r-1
        for n = 0:r-1
            push!(alpha, (s,t)->mk2d_1(t, s, m, n, e))
            push!(beta, (s,t)->mk2d_2(t, s, m, n, e))
        end
    end
    return f, alpha, beta
end

@testset "construct(separate)" begin
    Hparams.ConsMethod = "separate"
    n = 12
    h = 1/2^n
    Hparams.Geom = collect(0:2^n-1)*h    
    Hparams.Kernel, Hparams.α, Hparams.β = Merton_Kernel(1.0, 5)
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    e = maximum(abs.(FullMat(Hparams.Kernel, c.X, c.X)-Array(H)))
    @info "Error = $e" 
end

@testset "construct2(separate)" begin
    Hparams.ConsMethod = "separate"

    x = LinRange(0,1,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel, Hparams.α, Hparams.β = Merton_Kernel2D(1.0, 5)
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    e = maximum(abs.(FullMat(Hparams.Kernel, c.X, c.X)-Array(H)))
    @info "Error = $e" 
end