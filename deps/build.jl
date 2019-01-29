using CMake

cd("src")
run(`rm -rf build`)
run(`mkdir build`)
cd("build")
run(`$cmake ..`)
run(`make`)
cd("..")
cd("..")