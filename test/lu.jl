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
    Hparams.Kernel = (x,y)->1/(100+sum((x-y).^2))
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    H += 10I
    luH = lu(H)
    H = Array(H)
    P = luH.P
    luH = Array(luH)
    @test maximum(abs.(getl(luH)*getu(luH)-H[P,:]))<1e-8
end

@testset "solve2-dense" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(100+sum((x-y).^2))
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    H += 10I
    dH = FullMat(Hparams.Kernel, c.X, c.X)
    dH += 10I
    y = rand(2500)

    x1 = dH\y
    lH = lu(H)
    x2 = lH\y
    @info "Error after LU = $(norm(x1-x2)/norm(y))"

    x1 = dH\y
    x2 = Array(H)\y
    @info "Error before LU = $(norm(x1-x2)/norm(y))"
end



@testset "aca2" begin
    X = [LinRange(0,1,100)|>collect; LinRange(5,6,100)|>collect]
    U,V = aca2((x,y)->1/(1+(x-y)^2), X, X, 20)
    A = FullMat((x,y)->1/(1+(x-y)^2), X, X)
    maximum(abs.(A-U*V'))
end