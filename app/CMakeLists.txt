cmake_minimum_required(VERSION 3.5)

# Build the c++ functions into an executable
add_library(joptimtraj SHARED src/main/cpp/main.cpp)

#Add dependent libraries. Here ipopt is the only library
add_library(ipopt SHARED IMPORTED) # or STATIC instead of SHARED
set_property(TARGET ipopt PROPERTY IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libipopt.so")

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

#Tell the compiler that joptimtraj is dependent on ipopt
target_link_libraries(joptimtraj ipopt  ${log-lib} )

#Location of header files
include_directories(${CMAKE_SOURCE_DIR}/libs/include
                    ${CMAKE_SOURCE_DIR}/libs/include/ThirdParty)
