include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_stub test_stub.cpp)
target_link_libraries(test_stub
    PRIVATE
        GTest::gtest_main
)

# let header search path include test_stub.cpp
target_include_directories(test_stub
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(test_stub)