__precompile__(false)
module Hmatrix
    using Parameters
    using LinearAlgebra
    using PyPlot
    using Printf
    using Statistics
    using FastGaussQuadrature
    using PyCall
    using SparseArrays

    @pyimport numpy as np
    @pyimport scipy.sparse.linalg as ssl
    @pyimport sklearn.cluster as cluster
    @pyimport scipy.spatial as ss

    @pyimport matplotlib.patches as patch

    include("cluster.jl")
    include("harithm.jl")
    include("tools.jl")
    include("utils.jl")
    include("hlu.jl")
end
    


    