using Pkg

packages = [
    "Parameters",
    "LinearAlgebra",
    "PyPlot",
    "Printf",
    "Statistics",
    "Profile",
    "FFTW",
    "SpecialFunctions",
    "LowRankApprox",
    "FastGaussQuadrature",
    "TimerOutputs",
    "PyCall",
    "LinearAlgebra",
    "HCubature",
    "Polynomials",
    "ProgressMeter",
    "DelimitedFiles",
    "ToeplitzMatrices",
    "IterativeSolvers",
    "JLD2",
    "StatsFuns",
    "FileIO",
    "Interpolations"
]

# step 1: install packages
for p in packages
    Pkg.add(p)
    Pkg.build(p)
end

# step 2: build 
println("Please modify the CMakeLists.txt in third-party/ as needed if any error occurs")
try
    mkdir("./third-party/build")
catch
end
cd("./third-party/build")
try
    run(`cmake ..`)
    run(`make -j`)
catch
    println("Error Occured. Please modify the CMakeLists.txt and make sure you have g/g++ compiler and cmake installed.")
end

cd("../..")

# step 3: run a small test
include("test_Merton_kernel.jl")
test_case_2()
