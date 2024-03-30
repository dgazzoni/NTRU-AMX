cmake_minimum_required(VERSION 3.18)

project(ntru-AMX C CXX ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

option(CMAKE_UNITY_BUILD "merge source files into batches for faster compilation" ON)
option(SANITIZER "build with AddressSanitizer and UBSanitizer support" OFF)
option(BUILD_TESTING "build with tests enabled" ON)
option(USE_FEAT_DIT "enable device-independent timing bit" ON)

set(CMAKE_UNITY_BUILD_BATCH_SIZE 0)

if(BUILD_TESTING)
    include(CTest)
endif()

include(BuildCaching.cmake)
include(CreateSymlink.cmake)
include(CPUDetection.cmake)

add_compile_options(-Wall -Wextra -pedantic)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-language-extension-token -Wno-strict-prototypes)
endif()

if((CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15) OR
    CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wno-array-parameter)
endif()

if(SANITIZER)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

add_compile_options(-march=armv8-a+crypto)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        "SHELL:-mllvm -align-all-functions=6" "SHELL:-mllvm -align-all-nofallthru-blocks=6")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options(-falign-functions=64)
endif()

if(APPLE)
    execute_process(COMMAND brew --prefix openssl OUTPUT_VARIABLE OPENSSL_ROOT_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_package(OpenSSL REQUIRED)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
add_subdirectory(googletest)

include(GoogleTest)
set(GTEST_DISCOVERY_TIMEOUT 60)

add_library(ref_rng OBJECT reference/Reference_Implementation/crypto_kem/ntruhps2048509/rng.c)
target_compile_definitions(ref_rng PUBLIC randombytes_init=nist_randombytes_init randombytes=nist_randombytes)
target_compile_options(ref_rng PRIVATE -Wno-sign-compare -Wno-unused-parameter)

if((CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER 10) OR
    CMAKE_C_COMPILER_ID MATCHES "GNU")
    target_compile_options(ref_rng PRIVATE -Wno-unused-but-set-variable)
endif()

target_link_libraries(ref_rng PUBLIC OpenSSL::SSL OpenSSL::Crypto)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_library(opt_rng OBJECT rng_opt/rng_inline_asm.c)
else()
    add_library(opt_rng OBJECT rng_opt/rng.c)
    set_source_files_properties(rng_opt/rng.c PROPERTIES COMPILE_FLAGS -fno-strict-aliasing)
endif()

target_compile_definitions(opt_rng PUBLIC randombytes_init=opt_randombytes_init randombytes=opt_randombytes)
add_library(neon_rng ALIAS opt_rng)

set(REF_COMMON_SOURCES
    cmov.c fips202.c kem.c owcpa.c pack3.c packq.c poly.c poly_lift.c
    poly_mod.c poly_r2_inv.c poly_s3_inv.c poly_rq_mul.c sample.c sample_iid.c)

set(REF_HPS_SOURCES crypto_sort_int32.c)

set(PARAMETER_SETS hps2048509 hps2048677 hps4096821 hrss701)
set(KAT_NUMS 935 1234 1590 1450)

macro(ADD_KAT_TESTS)
    # https://stackoverflow.com/a/3071370/523079
    add_test(
        NAME ${LIBRARY}.KATs_match_spec
        COMMAND
        ${CMAKE_COMMAND}
        -DKATgen_cmd=${PQCGENKAT_KEM}
        -DKAT_expected=${CMAKE_SOURCE_DIR}/reference/KAT/ntru${PARAMETER_SET}/PQCkemKAT_${KAT_NUM}
        -DKAT_actual=PQCkemKAT_${KAT_NUM}
        -DWORKING_DIRECTORY=${CMAKE_BINARY_DIR}/KAT/${LIBRARY}
        -DBUILD_DIRECTORY=${CMAKE_BINARY_DIR}
        -DSRC_DIRECTORY=${CMAKE_SOURCE_DIR}
        -P ${CMAKE_SOURCE_DIR}/CompareKATs.cmake)
endmacro()

foreach(PARAMETER_SET KAT_NUM IN ZIP_LISTS PARAMETER_SETS KAT_NUMS)
    set(LIBRARY ref_ntru${PARAMETER_SET})

    set(PQCGENKAT_KEM PQCgenKAT_kem_${LIBRARY})

    add_library(${LIBRARY} STATIC)
    target_include_directories(${LIBRARY} PUBLIC
        reference/Reference_Implementation/crypto_kem/ntru${PARAMETER_SET})

    target_compile_options(${LIBRARY} PUBLIC -DCRYPTO_NAMESPACE\(s\)=ntru_\#\#s)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        target_compile_options(${LIBRARY} PRIVATE -Wno-stringop-overread)
    endif()

    target_link_libraries(${LIBRARY} PUBLIC ref_rng OpenSSL::SSL OpenSSL::Crypto)

    add_executable(${PQCGENKAT_KEM} reference/Reference_Implementation/crypto_kem/ntru${PARAMETER_SET}/PQCgenKAT_kem.c)
    target_compile_options(${PQCGENKAT_KEM} PRIVATE -Wno-unused-result)
    target_link_libraries(${PQCGENKAT_KEM} PRIVATE ${LIBRARY})

    if(CMAKE_UNITY_BUILD)
        set_target_properties(${LIBRARY} PROPERTIES UNITY_BUILD_MODE GROUP)
        set_target_properties(${PQCGENKAT_KEM} PROPERTIES UNITY_BUILD_MODE GROUP)
    endif()

    foreach(REF_COMMON_SOURCE ${REF_COMMON_SOURCES})
        target_sources(${LIBRARY} PRIVATE
            reference/Reference_Implementation/crypto_kem/ntru${PARAMETER_SET}/${REF_COMMON_SOURCE})
    endforeach()

    ADD_KAT_TESTS()

    if(NOT PARAMETER_SET STREQUAL hrss701)
        foreach(REF_HPS_SOURCE ${REF_HPS_SOURCES})
            target_sources(${LIBRARY} PRIVATE
                reference/Reference_Implementation/crypto_kem/ntru${PARAMETER_SET}/${REF_HPS_SOURCE})
        endforeach()
    endif()
endforeach()

set(HASH_PATH ${CMAKE_SOURCE_DIR}/vector-polymul-ntru-ntrup/hash)
set(HASH_SOURCES ${HASH_PATH}/fips202.c ${HASH_PATH}/sha2.c)

set(SORT_PATH ${CMAKE_SOURCE_DIR}/vector-polymul-ntru-ntrup/sort)
set(SORT_SOURCES ${SORT_PATH}/crypto_sort.c)

set(RAND_PATH ${CMAKE_SOURCE_DIR}/rng_opt)

set(AMX_PATH ${CMAKE_SOURCE_DIR}/amx)
set(AMX_SOURCES ${CMAKE_SOURCE_DIR}/amx/polymodmul.c ${CMAKE_SOURCE_DIR}/amx/aux_routines.c)

add_library(cycles OBJECT ${CMAKE_SOURCE_DIR}/vector-polymul-ntru-ntrup/cycles/m1cycles.c)

if(USE_FEAT_DIT)
    target_sources(cycles PRIVATE ${CMAKE_SOURCE_DIR}/speed/feat_dit.c)
    target_compile_definitions(cycles PUBLIC USE_FEAT_DIT)
endif()

target_include_directories(cycles PUBLIC ${CMAKE_SOURCE_DIR}/vector-polymul-ntru-ntrup/cycles ${CMAKE_SOURCE_DIR}/speed)

add_subdirectory(PQC_NEON/neon/ntru)
add_subdirectory(vector-polymul-ntru-ntrup)

# Tests
add_executable(test_rng test/test_rng.cpp)
target_include_directories(test_rng PRIVATE reference/Reference_Implementation/crypto_kem/ntruhps2048509)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_rng PRIVATE -Wno-macro-redefined)
endif()

target_link_libraries(test_rng PRIVATE ref_rng opt_rng GTest::gtest_main)
gtest_discover_tests(test_rng DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})

foreach(TEST_TYPE polymul polymodmul polymodmul_karatsuba)
    add_executable(test_${TEST_TYPE} test/test_${TEST_TYPE}.cpp amx/aux_routines.c amx/${TEST_TYPE}.c)

    if(TEST_TYPE MATCHES karatsuba)
        target_sources(test_${TEST_TYPE} PRIVATE amx/polymul.c)
    endif()

    target_include_directories(test_${TEST_TYPE} PRIVATE amx test)
    target_link_libraries(test_${TEST_TYPE} PRIVATE GTest::gtest_main)
    gtest_discover_tests(test_${TEST_TYPE} DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})
endforeach()

# Benchmarks
set(SPEED_MUL_NAMES polymul_ntru polymul_scaling polymodmul_ntru polymodmul_scaling)
set(SPEED_MUL_SRCS polymul.c polymul.c polymodmul.c polymodmul.c)

foreach(SPEED_MUL_NAME SPEED_MUL_SRC IN ZIP_LISTS SPEED_MUL_NAMES SPEED_MUL_SRCS)
    add_executable(speed_${SPEED_MUL_NAME} speed/speed_${SPEED_MUL_NAME}.c amx/aux_routines.c amx/${SPEED_MUL_SRC})
    target_include_directories(speed_${SPEED_MUL_NAME} PRIVATE ${AMX_PATH} ${RAND_PATH})
    target_link_libraries(speed_${SPEED_MUL_NAME} PRIVATE neon_rng cycles)
endforeach()

add_executable(speed_polymodmul_karatsuba_scaling
    speed/speed_polymodmul_scaling.c amx/aux_routines.c amx/polymodmul_karatsuba.c amx/polymul.c)
target_compile_definitions(speed_polymodmul_karatsuba_scaling PRIVATE KARATSUBA)
target_include_directories(speed_polymodmul_karatsuba_scaling PRIVATE ${AMX_PATH} ${RAND_PATH})
target_link_libraries(speed_polymodmul_karatsuba_scaling PRIVATE neon_rng cycles)

add_executable(speed_rng speed/speed_rng.c)
target_link_libraries(speed_rng PRIVATE ref_rng neon_rng cycles)

add_executable(speed_blas speed/speed_blas.c)
target_compile_definitions(speed_blas PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(speed_blas cycles "-framework Accelerate")

add_executable(latency_experiment speed/latency_experiment.c amx/latency_experiment.c)
target_include_directories(latency_experiment PRIVATE ${AMX_PATH} ${RAND_PATH})
target_link_libraries(latency_experiment PRIVATE neon_rng cycles)
