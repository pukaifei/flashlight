cmake_minimum_required(VERSION 3.5.1)

add_library(
  runtime_lpm
  INTERFACE
  )

target_sources(
  runtime_lpm
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/DataScheduler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Eval.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Init.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  runtime_lpm
  INTERFACE
  runtime # inherit all properties from base interface
  common
  data
  module
  )
