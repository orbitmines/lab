# embed_shader.cmake - Convert WGSL file to C++ raw string literal include
# Usage: cmake -DINPUT=foo.wgsl -DOUTPUT=foo.wgsl.inc -P embed_shader.cmake

file(READ "${INPUT}" CONTENT)
file(WRITE "${OUTPUT}" "R\"WGSL(\n${CONTENT})WGSL\"")
