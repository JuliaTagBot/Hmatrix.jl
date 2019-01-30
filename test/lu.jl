using LinearAlgebra
@testset "lu" begin
    X = LinRange(0,10,2000)|>collect
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 25
    c, H = NewHmat()
    luH = lu(H)
    H = Array(H)
    P = luH.P
    luH = Array(luH)
    @test maximum(abs.(getl(luH)*getu(luH)-H[P,:]))<1e-8
end

@testset "solve" begin
    X = LinRange(0,10,2000)|>collect
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 25
    c, H = NewHmat()
    H += 10I
    dH = Array(H) 
    y = rand(2000)
    x1 = dH\y
    lu!(H)
    x2 = H\y
    @test x1≈x2
end

@testset "lu2" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+norm(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 25
    c, H = NewHmat()
    H += 10I
    luH = lu(H)
    H = Array(H)
    P = luH.P
    luH = Array(luH)
    @test maximum(abs.(getl(luH)*getu(luH)-H[P,:]))<1e-8
end

@testset "solve2" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+norm(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 25
    c, H = NewHmat()
    H += 10I
    dH = Array(H) 
    y = rand(2500)
    x1 = dH\y
    lu!(H)
    x2 = H\y
    @test x1≈x2
end


