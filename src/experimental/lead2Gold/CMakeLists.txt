cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental-lead2gold)

# Build a version of the wav2letter++ object containing default wav2letter++ interfaces
# minus interface libraries which were custom defined, which are linked explicitly
add_library(
  wav2letter++_l2g
  ""
  )

# Add custom targets
add_subdirectory(${PROJECT_SOURCE_DIR}/src/common) # for target common_l2g
add_subdirectory(${PROJECT_SOURCE_DIR}/src/criterion) # for target criterion_l2g
add_subdirectory(${PROJECT_SOURCE_DIR}/src/data) # for target data_l2g
#add_subdirectory(${PROJECT_SOURCE_DIR}/src/libraries/common) # for target common-library_l2g
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime) # for target runtime_l2g


target_link_libraries(
  wav2letter++_l2g
  PUBLIC
  wav2letter++
  common_l2g      #l2g-specific
  criterion_l2g   #l2g-specific
  data_l2g        #l2g-specific
#common-library_l2g  #l2g-specific
  runtime_l2g     #l2g-specific
  )





# ------------------------- Dedicated l2g Train executable --------------------------
add_executable(
  Train_l2g
  Train.cpp
  )

target_link_libraries(
  Train_l2g
  wav2letter++_l2g
)

add_executable(
  Test_l2g
  Test.cpp
  )

target_link_libraries(
  Test_l2g
  wav2letter++_l2g
  )

add_executable(
  Decoder_l2g
  Decode.cpp
  )

target_link_libraries(
  Decoder_l2g
  wav2letter++_l2g
  )


  # ----------------------------- Custom executable -----------------------------

add_executable(
  SaveExample
  SaveExample.cpp
)

target_link_libraries(
  SaveExample
  wav2letter++_l2g
)

add_executable(
  ComputeStats
  ComputeStats.cpp
)

target_link_libraries(
  ComputeStats
  wav2letter++_l2g
)

add_executable(
  ComputewLER
  ComputewLER.cpp
)

target_link_libraries(
  ComputewLER
  wav2letter++_l2g
)

add_executable(
  FindScale1
  FindScale1.cpp
)

target_link_libraries(
  FindScale1
  wav2letter++_l2g
)

add_executable(
  FindScale2
  FindScale2.cpp
)

target_link_libraries(
  FindScale2
  wav2letter++_l2g
)

add_executable(
  FindScale3
  FindScale3.cpp
)

target_link_libraries(
  FindScale3
  wav2letter++_l2g
)

add_executable(
  UpdateNoiselm
  UpdateNoiselm.cpp
)

target_link_libraries(
  UpdateNoiselm
  wav2letter++_l2g
)

add_executable(
  UpdateNoiselmNN
  UpdateNoiselmNN.cpp
)

target_link_libraries(
  UpdateNoiselmNN
  wav2letter++_l2g
)

add_executable(
  UpdateNoiselmNN_withX
  UpdateNoiselmNN_withX.cpp
)

target_link_libraries(
  UpdateNoiselmNN_withX
  wav2letter++_l2g
)

add_executable(
  Train_ss
  Train_ss.cpp
)

target_link_libraries(
  Train_ss
  wav2letter++_l2g
)


# Tests
if (1)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/tests)
endif ()