cmake_minimum_required(VERSION 3.5.1)

add_library(
  runtime_l2g
  INTERFACE
  )

target_sources(
  runtime_l2g
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Logger.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Init.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DataScheduler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Eval.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.cpp
  )

target_link_libraries(
  runtime_l2g
  INTERFACE
  runtime # inherit all properties from base interface
  )
