include(FetchContent)

# ── godot-cpp via FetchContent ──────────────────────────────────────────────

FetchContent_Declare(
    godot-cpp
    GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
    GIT_TAG        godot-4.5-stable
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(godot-cpp)

# ── GDExtension shared library ──────────────────────────────────────────────

add_library(compression_gdextension SHARED
    compression_extension.cpp
    register_types.cpp
)

target_link_libraries(compression_gdextension
    godot-cpp
    compression_estimator
)

target_include_directories(compression_gdextension PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Match Godot's naming convention
set_target_properties(compression_gdextension PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "compression_gdextension"
)
