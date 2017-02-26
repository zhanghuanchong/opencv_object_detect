cmake_minimum_required(VERSION 3.6)
project(opencv_test)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(opencv_test ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(opencv_test ${OpenCV_LIBS})