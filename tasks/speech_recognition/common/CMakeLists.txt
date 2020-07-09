cmake_minimum_required(VERSION 3.5.1)

add_library(
  asr-common
  INTERFACE
  )

target_sources(
  asr-common
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  )

target_link_libraries(
  asr-common
  INTERFACE
  flashlight
  ${GFLAGS_LIBRARIES}
  )

target_include_directories(
  asr-common
  INTERFACE
  ${CMAKE_SOURCE_DIR}
  ${GFLAGS_INCLUDE_DIRS}
  )
