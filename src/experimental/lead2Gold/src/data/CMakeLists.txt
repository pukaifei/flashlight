cmake_minimum_required(VERSION 3.5.1)

add_library(
  data_l2g
  INTERFACE
  )

target_sources(
  data_l2g
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Featurize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NoiseW2lListFilesDataset.cpp
  )

target_link_libraries(
  data_l2g
  INTERFACE
  data # inherit all properties from base interface
  )
