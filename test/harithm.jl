@testset "matmul" begin
    X = LinRange(0,10,2000)|>collect
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(1+(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 5
    c, H = NewHmat()
    A = FullMat(Hparams.Kernel, c.X, c.X)
    y = rand(2000)
    @test A*y ≈ H*y
end

@testset "matmul2" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(100+norm(x-y)^2)
    Hparams.MaxBlock = 500
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    A = FullMat(Hparams.Kernel, c.X, c.X)
    y = rand(2500)
    @test A*y ≈ H*y
end