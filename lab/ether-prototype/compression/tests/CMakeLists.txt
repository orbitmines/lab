add_executable(test_cpu_reference test_cpu_reference.cpp)
target_link_libraries(test_cpu_reference compression_estimator)
target_include_directories(test_cpu_reference PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Find enwik8 relative to source tree
set(ENWIK8_PATH "${PROJECT_SOURCE_DIR}/../enwik8")

add_test(NAME cpu_reference COMMAND test_cpu_reference "${ENWIK8_PATH}")

# GPU kernel tests
add_executable(test_gpu_kernels test_gpu_kernels.cpp)
target_link_libraries(test_gpu_kernels compression_estimator)
target_include_directories(test_gpu_kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME gpu_kernels COMMAND test_gpu_kernels "${ENWIK8_PATH}")

# GPU verification tests (comprehensive)
add_executable(test_gpu_verification test_gpu_verification.cpp)
target_link_libraries(test_gpu_verification compression_estimator)
target_include_directories(test_gpu_verification PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME gpu_verification COMMAND test_gpu_verification "${ENWIK8_PATH}")

# GPU session tests
add_executable(test_gpu_session test_gpu_session.cpp)
target_link_libraries(test_gpu_session compression_estimator)
target_include_directories(test_gpu_session PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME gpu_session COMMAND test_gpu_session "${ENWIK8_PATH}")

# Benchmark
add_executable(bench_enwik bench_enwik.cpp)
target_link_libraries(bench_enwik compression_estimator)
target_include_directories(bench_enwik PRIVATE ${PROJECT_SOURCE_DIR}/include)
