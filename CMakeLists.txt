cmake_minimum_required(VERSION 3.10)
project(EdgeLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3") # Maximize speed

add_executable(edge_llm src/main.cpp)