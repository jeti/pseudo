cmake_minimum_required(VERSION 3.5)

# If a build type is not passed to cmake, then use this...
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    # set(CMAKE_BUILD_TYPE Debug)
endif()

# If you are debugging, generate symbols.
# See here for how to get ASAN line numbers:
# https://tsdgeos.blogspot.de/2014/03/asan-and-gcc-how-to-get-line-numbers-in.html
# set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -fstack-protector-all")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# If in release mode, use all possible optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# We need c++11
set(CMAKE_CXX_STANDARD 14)

# Show us all of the warnings and enable all vectorization options!!!
# I must be crazy because these vectorization flags seem to have no effect.
set(CMAKE_CXX_FLAGS "-Wall -mavx -mfma")

if (CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
    include_directories(../../Eigen/
            /cygdrive/c/coin/windows/ipopt/include/coin/
            /cygdrive/c/coin/windows/ipopt/include/coin/ThirdParty/)
    find_library(IPOPT_LIBRARY ipopt HINTS /cygdrive/c/coin/windows/ipopt/lib/)
else ()
    include_directories(../../Eigen/
            ../../coin/CoinIpopt/build/include/coin/
            ../../coin/CoinIpopt/build/include/coin/ThirdParty/)
    find_library(IPOPT_LIBRARY ipopt HINTS ../../coin/CoinIpopt/build/lib/)
endif ()

# Build the c++ functions into an executable
add_executable(trajectory_optimization main.cpp cout.h)

# Link all of the libraries together so that the C++-executable can call IPOPT
target_link_libraries(trajectory_optimization ${IPOPT_LIBRARY})