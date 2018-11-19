cmake_minimum_required(VERSION 3.5.1)
project(ifmm)
#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -g -msse2")
link_directories(/usr/)
include_directories(. /home/kailaix/.local/include/eigen3)
add_executable(main ACA_FullyPivoted.cpp)
add_library(aca SHARED ACA_FullyPivoted.cpp)
set_property(TARGET aca PROPERTY POSITION_INDEPENDENT_CODE ON)


add_executable(lr lowrank.cpp)

target_link_libraries(main lapack blas m)

