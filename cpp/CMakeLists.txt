cmake_minimum_required(VERSION 3.20)
project(deeplob_hft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O3 -DNDEBUG)

# ---- ONNX Runtime paths (您本机真实路径) ----
set(ORT_INCLUDE_DIR "/opt/homebrew/opt/onnxruntime/include/onnxruntime")
# set(ORT_LIBRARY     "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib")
set(ORT_LIBRARY     "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.1.22.2.dylib")

# 做成一个 imported target：onnxruntime::onnxruntime
add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)

add_library(core
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

# core 里用到了 Ort::，所以让 core 直接链接 ORT（可执行文件会继承到）
target_link_libraries(core
  PUBLIC
    onnxruntime::onnxruntime
)

add_executable(main_stream src/main_stream.cpp)
target_link_libraries(main_stream PRIVATE core)

add_executable(parity_run src/parity_run.cpp)
target_link_libraries(parity_run PRIVATE core)
