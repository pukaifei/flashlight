cmake_minimum_required(VERSION 3.5.1)

function(build_script SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/..
    )
endfunction(build_script)

if (W2L_BUILD_SCRIPTS)
  build_script(${CMAKE_SOURCE_DIR}/scripts/Seq2SeqAnalysis.cpp)
  build_script(${CMAKE_SOURCE_DIR}/scripts/LmAnalysis.cpp)
endif ()
