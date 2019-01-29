using CMake

run(`cd src && rm -rf build && mkdir build && cd build && $cmake .. && make && cd .. && cd ..`)
