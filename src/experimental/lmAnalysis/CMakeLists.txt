cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental-lmAnalysis)

# --------------------------- LMAnalysis -------------------------------
add_executable(
  LMAnalysis
  LMAnalysis.cpp
)

target_link_libraries(
  LMAnalysis
  wav2letter++_lpm
)

# ----------------------------- DumpBeamSearch -----------------------------
add_executable(
	DumpBeamSearch
	DumpBeamSearch.cpp
)

target_link_libraries(
	DumpBeamSearch
  wav2letter++_lpm
)

# --------------------------- DumpBeamSearchWithLm -------------------------------
add_executable(
  DumpBeamSearchWithLm
  DumpBeamSearchWithLm.cpp
)

target_link_libraries(
  DumpBeamSearchWithLm
  wav2letter++_lpm
)
