cmake_minimum_required(VERSION 3.15)
project(FastReplay VERSION 0.1.0 LANGUAGES CXX)

# c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find pybind11 (auto-detect from pip/conda install)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# 2. pybind11 module
pybind11_add_module(fastreplay src/fastreplay.cpp)

target_include_directories(fastreplay
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# install target (for pip install / scikit-build-core)
install(TARGETS fastreplay DESTINATION .)

# Tests (optional, use -DBUILD_TESTING=ON to enable)
option(BUILD_TESTING "Build C++ tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()