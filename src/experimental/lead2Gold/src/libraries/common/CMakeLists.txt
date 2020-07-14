cmake_minimum_required(VERSION 3.5.1)

add_library(
  common-library_l2g
  INTERFACE
  )

target_sources(
  common-library_l2g
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Dictionary.cpp
  )

target_link_libraries(
  common-library_l2g
  INTERFACE
  common-library # inherit all properties from base interface
  )
