cmake_minimum_required(VERSION 3.5.1)

# Main library
add_library(
  criterion_l2g
  INTERFACE
  )

# ---------------------------- lead2gold -----------------------------

target_sources(
  criterion_l2g
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/NoiseTrie.cpp
#${CMAKE_CURRENT_SOURCE_DIR}/l2g/AnalyseBeam.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/AutoSegBeamNoiseCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/CtcBeamNoiseCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/CtcForceAlignBeamNoise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/ForceAlignBeamNoise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/ForceAlignBeamNoiseStats.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/NoiseLMLetterSwapPretrain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/EncDecCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/EncDecCriterionMulti.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/TransformerBlockSimple.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/TransformerBlockAttend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/l2g/TransformerBlockMultiAttend.cpp
  )


target_link_libraries(
  criterion_l2g
  INTERFACE
  )

#target_include_directories(
#  lead2gold
#  INTERFACE
#  #lead2gold
#  src/flashlight
#  ${cereal_INCLUDE_DIRS}
#  )
#
#target_link_libraries(
#  lead2gold
#  INTERFACE
#  flashlight::flashlight
#  common
#  ${cereal_LIBRARIES}
#  )
