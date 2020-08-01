cmake_minimum_required(VERSION 3.5.1)

set(
  MEMORY_MANAGER_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/AllocationLog.cpp
  ${CMAKE_CURRENT_LIST_DIR}/allocator/CompositeMemoryAllocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/allocator/ConfigurableMemoryAllocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/allocator/MemoryAllocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/allocator/freelist/FreeList.cpp
  ${CMAKE_CURRENT_LIST_DIR}/allocator/memorypool/MemoryPool.cpp
  ${CMAKE_CURRENT_LIST_DIR}/optimizer/Optimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/optimizer/Simulator.cpp
)

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${MEMORY_MANAGER_SOURCES}>
)

add_executable(allocation_log_analyser
  ${CMAKE_CURRENT_LIST_DIR}/AllocationLogAnalyser.cpp
  )
add_executable(stand_alone_simulator
  ${CMAKE_CURRENT_LIST_DIR}/StandAloneSimulator.cpp
)
add_executable(stand_alone_optimizer
  ${CMAKE_CURRENT_LIST_DIR}/StandAloneOptimizer.cpp
  )

target_link_libraries(allocation_log_analyser flashlight)
target_link_libraries(stand_alone_simulator flashlight)
target_link_libraries(stand_alone_optimizer flashlight)
