using Hmatrix
using Test
using PyCall
using LinearAlgebra
@pyimport numpy as np

Hparams.verbose = true
Hparams.CompMethod = "bbfmm"
include("cluster.jl")
include("harithm.jl")
include("lu.jl")

