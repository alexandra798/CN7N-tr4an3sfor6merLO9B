cmake_minimum_required(VERSION 3.20)
project(deeplob_hft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -DNDEBUG)
  message(STATUS "Build type: Release (optimized)")
else()
  add_compile_options(-O0 -g)
  message(STATUS "Build type: Debug")
endif()

# ---- ONNX Runtime paths ----
# Adjust these paths to match your system
set(ORT_INCLUDE_DIR "/opt/homebrew/opt/onnxruntime/include/onnxruntime")
set(ORT_LIBRARY     "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.1.22.2.dylib")

# Alternative: use environment variable
# set(ORT_INCLUDE_DIR "$ENV{ONNX_RUNTIME_INCLUDE}")
# set(ORT_LIBRARY     "$ENV{ONNX_RUNTIME_LIB}")

if(NOT EXISTS ${ORT_INCLUDE_DIR})
  message(FATAL_ERROR "ONNX Runtime include not found: ${ORT_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${ORT_LIBRARY})
  message(FATAL_ERROR "ONNX Runtime library not found: ${ORT_LIBRARY}")
endif()

# Create imported target for ONNX Runtime
add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)

# ---- Core library ----
add_library(core STATIC
  src/session.cpp
  src/preproc.cpp
  src/fe_ofi.cpp
  src/litcvg_session.cpp
  src/tca.cpp
  src/npy.cpp
)

target_include_directories(core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(core
  PUBLIC
    onnxruntime::onnxruntime
)

# Enable threading support
find_package(Threads REQUIRED)
target_link_libraries(core PUBLIC Threads::Threads)

# ---- Executables ----

# Original streaming inference
add_executable(main_stream src/main_stream.cpp)
target_link_libraries(main_stream PRIVATE core)

# Parity check (Python vs C++)
add_executable(parity_run src/parity_run.cpp)
target_link_libraries(parity_run PRIVATE core)

# ⭐ NEW: Threaded pipeline
add_executable(main_stream_threaded src/main_stream_threaded.cpp)
target_link_libraries(main_stream_threaded PRIVATE core)

# ⭐ NEW: Performance benchmark suite
add_executable(perf_benchmark src/perf_benchmark.cpp)
target_link_libraries(perf_benchmark PRIVATE core)

# ---- Installation ----
install(TARGETS main_stream main_stream_threaded perf_benchmark parity_run
  RUNTIME DESTINATION bin
)

# ---- Summary ----
message(STATUS "")
message(STATUS "========================================")
message(STATUS "DeepLOB HFT Build Configuration")
message(STATUS "========================================")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime:      ${ORT_LIBRARY}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - main_stream           (single-thread)")
message(STATUS "  - main_stream_threaded  (3-thread pipeline) ⭐")
message(STATUS "  - perf_benchmark        (performance tests) ⭐")
message(STATUS "  - parity_run            (C++/Python parity)")
message(STATUS "========================================")
message(STATUS "")
