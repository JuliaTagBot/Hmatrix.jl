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
    @pyimport scipy.signal as ss
    @pyimport scipy.sparse.linalg as ssl
    @pyimport scipy.stats as sstats
    @pyimport scipy.special as ssp  
    @pyimport sklearn.cluster as cluster

    @pyimport matplotlib.patches as patch

    include("cluster.jl")
    include("harithm.jl")
    include("tools.jl")
    include("utils.jl")
    include("hlu.jl")
end
    


    