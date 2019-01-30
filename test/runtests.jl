using Hmatrix
using Test
using PyCall
using LinearAlgebra
@pyimport numpy as np

Hparams.verbose = true
include("cluster.jl")
include("harithm.jl")
include("lu.jl")
include("tools.jl")


