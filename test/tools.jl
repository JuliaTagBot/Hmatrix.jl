@testset "iterative solver" begin
    x = LinRange(0,10,50)
    X, Y = np.meshgrid(x, x)
    X = [X[:] Y[:]]
    Hparams.Geom = X
    Hparams.Kernel = (x,y)->1/(10+sum((x-y).^2))
    Hparams.MaxBlock = 256
    Hparams.MinBlock = 64
    Hparams.MaxRank = 15
    Hparams.ConsMethod = "svd"
    c, H = NewHmat()
    H += 10I
    luH = lu(H)

    densH = FullMat(Hparams.Kernel, c.X, c.X)
    densH += 10I

    y = rand(2500)
    x0 = densH \ y
    __cnt = 0
    function callback(rk)
        global __cnt
        println("Iter $__cnt, Residual $(norm(rk))")
        __cnt += 1
    end
    x = gmres(H, y;  callback=callback)
    e = norm(x-x0)/norm(x0)
    @info "Error = $e"
end