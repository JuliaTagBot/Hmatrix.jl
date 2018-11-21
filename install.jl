using Pkg

packages = [
    "Parameters",
    "LinearAlgebra",
    "PyPlot",
    "Printf",
    "Statistics",
    "Profile",
    "LowRankApprox",
    "FastGaussQuadrature",
    "TimerOutputs",
    "PyCall",
    "LinearAlgebra"
]

# step 1: install packages
for p in packages
    Pkg.add(p)
    Pkg.build(p)
end

# step 2: build 
println("Please change the CMakeLists.txt in third-party/ as needed if any error occurs")
cd("third-party/build")
run(`cmake ..`)
run(`make -j`)
cd("../..")

# step 3: run a small test
include("test_Merton_kernel.jl")
test_case_2()
