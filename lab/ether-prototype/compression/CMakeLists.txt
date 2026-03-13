cmake_minimum_required(VERSION 3.20)
project(compression_estimator VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# ── Dependencies ─────────────────────────────────────────────────────────────

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(ZLIB REQUIRED IMPORTED_TARGET zlib)
pkg_check_modules(LZ4 IMPORTED_TARGET liblz4)

find_library(DEFLATE_LIB deflate)
find_path(DEFLATE_INCLUDE libdeflate.h)

# ── WebGPU backend selection ─────────────────────────────────────────────────

set(WEBGPU_BACKEND "wgpu-native" CACHE STRING "WebGPU backend: wgpu-native (default), dawn, or emscripten")
set_property(CACHE WEBGPU_BACKEND PROPERTY STRINGS "wgpu-native" "dawn" "emscripten")

# Auto-detect Emscripten
if(EMSCRIPTEN AND NOT WEBGPU_BACKEND STREQUAL "emscripten")
    set(WEBGPU_BACKEND "emscripten" CACHE STRING "" FORCE)
    message(STATUS "Emscripten detected — auto-selecting emscripten WebGPU backend")
endif()

set(WGPU_NATIVE_VERSION "v27.0.4.0" CACHE STRING "wgpu-native release version")

if(WEBGPU_BACKEND STREQUAL "wgpu-native")
    # ── wgpu-native via FetchContent (pre-built release binary) ──────────────

    # Determine platform/arch for the download URL
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(_WGPU_OS "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_WGPU_OS "macos")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(_WGPU_OS "windows")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(_WGPU_OS "android")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_WGPU_OS "ios")
    else()
        message(FATAL_ERROR "Unsupported platform for wgpu-native: ${CMAKE_SYSTEM_NAME}")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(_WGPU_ARCH "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(_WGPU_ARCH "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
        set(_WGPU_ARCH "armv7")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
        set(_WGPU_ARCH "i686")
    else()
        message(FATAL_ERROR "Unsupported arch for wgpu-native: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Windows MSVC vs GNU
    if(_WGPU_OS STREQUAL "windows")
        if(MSVC)
            set(_WGPU_PLATFORM "${_WGPU_OS}-${_WGPU_ARCH}-msvc")
        else()
            set(_WGPU_PLATFORM "${_WGPU_OS}-${_WGPU_ARCH}-gnu")
        endif()
    else()
        set(_WGPU_PLATFORM "${_WGPU_OS}-${_WGPU_ARCH}")
    endif()

    set(_WGPU_URL "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}/wgpu-${_WGPU_PLATFORM}-release.zip")
    message(STATUS "Fetching wgpu-native ${WGPU_NATIVE_VERSION} from ${_WGPU_URL}")

    FetchContent_Declare(
        wgpu-native
        URL "${_WGPU_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(wgpu-native)

    set(_WGPU_DIR "${wgpu-native_SOURCE_DIR}")

    # Find the library — prefer static to avoid RPATH issues
    if(EXISTS "${_WGPU_DIR}/lib/libwgpu_native.a")
        set(_WGPU_LIB "${_WGPU_DIR}/lib/libwgpu_native.a")
        set(_WGPU_TYPE STATIC)
    elseif(EXISTS "${_WGPU_DIR}/lib/wgpu_native.lib")
        set(_WGPU_LIB "${_WGPU_DIR}/lib/wgpu_native.lib")
        set(_WGPU_TYPE STATIC)
    elseif(EXISTS "${_WGPU_DIR}/lib/libwgpu_native.so")
        set(_WGPU_LIB "${_WGPU_DIR}/lib/libwgpu_native.so")
        set(_WGPU_TYPE SHARED)
    elseif(EXISTS "${_WGPU_DIR}/lib/wgpu_native.dll.lib")
        set(_WGPU_LIB "${_WGPU_DIR}/lib/wgpu_native.dll.lib")
        set(_WGPU_TYPE SHARED)
    else()
        message(FATAL_ERROR "wgpu-native library not found in ${_WGPU_DIR}/lib/")
    endif()

    add_library(wgpu_native ${_WGPU_TYPE} IMPORTED)
    set_target_properties(wgpu_native PROPERTIES
        IMPORTED_LOCATION "${_WGPU_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_WGPU_DIR}/include"
    )
    set(HAS_WGPU TRUE)
    set(WEBGPU_LINK_TARGET wgpu_native)

elseif(WEBGPU_BACKEND STREQUAL "dawn")
    # ── Dawn (fetched via FetchContent) ──────────────────────────────────────

    # Dawn toggles — disable components we don't need to speed up the build.
    set(DAWN_FETCH_DEPENDENCIES ON  CACHE BOOL "" FORCE)
    set(DAWN_BUILD_SAMPLES      OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D11       OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D12       OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_NULL        OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_VULKAN      ON  CACHE BOOL "" FORCE)

    FetchContent_Declare(
        dawn
        GIT_REPOSITORY https://dawn.googlesource.com/dawn
        GIT_TAG        chromium/7012
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    message(STATUS "Fetching Dawn (this may take a while on first configure)...")
    FetchContent_MakeAvailable(dawn)

    set(HAS_WGPU TRUE)
    set(WEBGPU_LINK_TARGET webgpu_dawn)
    message(STATUS "Dawn backend enabled")

elseif(WEBGPU_BACKEND STREQUAL "emscripten")
    # ── Emscripten (browser-native WebGPU) ───────────────────────────────────
    # Emscripten provides webgpu.h natively; no external library needed.
    # The linker flag -sUSE_WEBGPU enables it.
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "emscripten backend requires building with emcmake/emcc")
    endif()
    set(HAS_WGPU TRUE)
    set(WEBGPU_LINK_TARGET "")
    set(EMSCRIPTEN_WEBGPU TRUE)
    message(STATUS "Emscripten WebGPU backend enabled (browser-native)")

else()
    message(FATAL_ERROR "Unknown WEBGPU_BACKEND: ${WEBGPU_BACKEND}. Use 'wgpu-native', 'dawn', or 'emscripten'.")
endif()

# ── Embed WGSL shaders as C string literals ──────────────────────────────────

if(HAS_WGPU)
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders")
    set(SHADER_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_INC_DIR}")

    file(GLOB WGSL_FILES "${SHADER_DIR}/*.wgsl")
    foreach(WGSL_FILE ${WGSL_FILES})
        get_filename_component(WGSL_NAME ${WGSL_FILE} NAME)
        set(INC_FILE "${SHADER_INC_DIR}/${WGSL_NAME}.inc")
        # Convert WGSL to a C raw string literal
        add_custom_command(
            OUTPUT "${INC_FILE}"
            COMMAND ${CMAKE_COMMAND}
                -DINPUT=${WGSL_FILE}
                -DOUTPUT=${INC_FILE}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shader.cmake
            DEPENDS "${WGSL_FILE}"
            COMMENT "Embedding ${WGSL_NAME}"
        )
        list(APPEND SHADER_INC_FILES "${INC_FILE}")
    endforeach()
    add_custom_target(embed_shaders DEPENDS ${SHADER_INC_FILES})
endif()

# ── Core shared library ─────────────────────────────────────────────────────

set(CORE_SOURCES
    src/estimator.cpp
    src/gpu_context.cpp
    src/gpu_session.cpp
    src/verification.cpp
    src/cpu/entropy_cpu.cpp
    src/cpu/huffman_cpu.cpp
    src/cpu/arithmetic_cpu.cpp
    src/cpu/ans_cpu.cpp
    src/cpu/reference_wrappers.cpp
    src/neural_compressor.cpp
    src/neural_train_session.cpp
)

if(HAS_WGPU)
    list(APPEND CORE_SOURCES src/gpu/webgpu_backend.cpp)
endif()

add_library(compression_estimator SHARED ${CORE_SOURCES})

target_include_directories(compression_estimator
    PUBLIC include
    PRIVATE src
)

if(HAS_WGPU)
    target_include_directories(compression_estimator PRIVATE "${SHADER_INC_DIR}")
    # The .inc files are in a "shaders" subdir of the binary dir
    target_include_directories(compression_estimator PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(WEBGPU_LINK_TARGET)
        target_link_libraries(compression_estimator ${WEBGPU_LINK_TARGET})
    endif()
    if(EMSCRIPTEN_WEBGPU)
        target_link_options(compression_estimator PUBLIC -sUSE_WEBGPU=1 -sASYNCIFY=1)
        target_compile_options(compression_estimator PRIVATE -sUSE_WEBGPU=1)
    endif()
    target_compile_definitions(compression_estimator PRIVATE HAS_WEBGPU=1)
    add_dependencies(compression_estimator embed_shaders)
endif()

target_link_libraries(compression_estimator
    PkgConfig::ZSTD
    PkgConfig::ZLIB
)

if(DEFLATE_LIB AND DEFLATE_INCLUDE)
    target_link_libraries(compression_estimator ${DEFLATE_LIB})
    target_include_directories(compression_estimator PRIVATE ${DEFLATE_INCLUDE})
else()
    message(WARNING "libdeflate not found — deflate_exact will be unavailable")
endif()

if(LZ4_FOUND)
    target_link_libraries(compression_estimator PkgConfig::LZ4)
    target_compile_definitions(compression_estimator PRIVATE HAS_LZ4=1)
    message(STATUS "LZ4 found: ${LZ4_VERSION}")
else()
    message(WARNING "liblz4 not found — lz4 estimates will be unavailable")
endif()

# ── Tests ────────────────────────────────────────────────────────────────────

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── GDExtension ──────────────────────────────────────────────────────────────

option(BUILD_GDEXTENSION "Build Godot GDExtension" OFF)
if(BUILD_GDEXTENSION)
    add_subdirectory(gdextension)
endif()
