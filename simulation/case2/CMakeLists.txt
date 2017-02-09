cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(ideal)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (ideal ideal.cpp)
target_link_libraries (ideal ${PCL_LIBRARIES})
