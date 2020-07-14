cmake_minimum_required(VERSION 3.5.1)

add_library(
  common_l2g
  INTERFACE
  )

target_sources(
  common_l2g
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  common_l2g
  INTERFACE
  common # inherit all properties from base interface
  )
