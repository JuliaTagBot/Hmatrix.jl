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
    Hparams.Kernel = (x,y)->1/(1+sum((x-y).^2))
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 3
    c, H = NewHmat()
    @test maximum(abs.(FullMat(Hparams.Kernel, c.X, c.X)-Array(H)))<1e-8
end
