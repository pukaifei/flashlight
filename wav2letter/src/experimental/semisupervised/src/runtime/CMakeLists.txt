cmake_minimum_required(VERSION 3.5.1)

add_library(
  runtime_ss
  INTERFACE
  )

target_sources(
  runtime_ss
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/DataScheduler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Eval.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Init.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  runtime_ss
  INTERFACE
  runtime # inherit all properties from base interface
  common
  data
  module
  )
