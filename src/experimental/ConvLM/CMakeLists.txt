cmake_minimum_required(VERSION 3.5.1)

add_library(
  convlm-experimental
  INTERFACE
  )

target_sources(
  convlm-experimental
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  convlm-experimental
  INTERFACE
  flashlight::flashlight
  module
  common
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  convlm-experimental
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  )
