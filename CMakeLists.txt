cmake_minimum_required(VERSION 3.18)
project(whisper_ultra_engine CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG v1.7.1
    GIT_SHALLOW TRUE
)

set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "" FORCE)

FetchContent_MakeAvailable(whisper)

add_executable(whisper_ultra_engine whisper_ultra_engine.cpp)
target_include_directories(whisper_ultra_engine PRIVATE
    ${whisper_SOURCE_DIR}/include
    ${whisper_SOURCE_DIR}/ggml/include
)
target_link_libraries(whisper_ultra_engine PRIVATE whisper)
