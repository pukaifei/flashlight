cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight)

if (FL_BUILD_TESTS)
  build_test(${DIR}/FreeListTest.cpp ${LIBS} "")
  build_test(${DIR}/MemoryPoolTest.cpp ${LIBS} "")
  build_test(${DIR}/CompositeMemoryAllocatorTest.cpp ${LIBS} "")
  build_test(${DIR}/ConfigurableMemoryAllocatorTest.cpp ${LIBS} "")
endif ()
