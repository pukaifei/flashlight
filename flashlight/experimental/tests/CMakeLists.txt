cmake_minimum_required(VERSION 3.5.1)

set(DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBS flashlight experimental_memory) # TODO: what is experimental_memory?? nuke it

if (FL_BUILD_TESTS)
  build_test(${DIR}/experimental/tests/FreeListTest.cpp ${LIBS} "")
  build_test(${DIR}/experimental/tests/MemoryPoolTest.cpp ${LIBS} "")
  build_test(${DIR}/experimental/tests/CompositeMemoryAllocatorTest.cpp ${LIBS} "")
  build_test(${DIR}/experimental/tests/ConfigurableMemoryAllocatorTest.cpp ${LIBS} "")
endif ()
