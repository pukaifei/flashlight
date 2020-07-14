cmake_minimum_required(VERSION 3.5.1)

if (W2L_BUILD_TESTS)
  # Criterion
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/test/CriterionTest_L2G.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/test/CriterionTest_DEBUG_L2G.cpp)
endif ()
