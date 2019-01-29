@testset "bisect_cluster" begin
    X = rand(100,2)
    X1, P1, X2, P2 = bisect_cluster(X)
end