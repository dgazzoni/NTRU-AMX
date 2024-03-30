set(SOURCES_UNITY cmov.c kem.c neon_poly_mod.c owcpa.c pack3.c packq.c poly_r2_inv.c poly.c sample.c)
set(SOURCES_NO_UNITY neon_poly_lift.c poly_s3_inv.c)

set(SOURCES_NTRU_OPT neon_batch_multiplication.c neon_matrix_transpose.c neon_poly_rq_mul.c)
set(SOURCES_NTRU_AMX amx_poly_rq_mul.c)

set(SOURCES_hps2048509 neon_sample_iid.c)
set(SOURCES_hps2048677 sample_iid.c)
set(SOURCES_hps4096821 sample_iid.c)
set(SOURCES_hrss701 sample_iid.c)

set(SPEED_PREFIXES speed speed_polymul speed_polymul_ct)
set(SPEED_SOURCES speed.c speed_polymul.c speed_polymul_ct.c)
set(SPEED_NTESTSS 1024 1024 64)

foreach(PARAMETER_SET KAT_NUM IN ZIP_LISTS PARAMETER_SETS KAT_NUMS)
    foreach(ALLOC stack mmap)
        foreach(IMPL neon amx)
            set(LIBRARY ntru${PARAMETER_SET}_${ALLOC}_NG21_${IMPL})
            set(PQCGENKAT_KEM PQCgenKAT_kem_${LIBRARY})

            add_library(${LIBRARY} STATIC ${HASH_SOURCES})
            target_compile_options(${LIBRARY} PUBLIC -DCRYPTO_NAMESPACE\(s\)=ntru_\#\#s)
            target_link_libraries(${LIBRARY} PUBLIC neon_rng)

            if(IMPL STREQUAL neon)
                set(SOURCES_IMPL ${SOURCES_NTRU_OPT})
            else()
                set(SOURCES_IMPL ${SOURCES_NTRU_AMX})
                target_sources(${LIBRARY} PRIVATE ${AMX_SOURCES})
                target_include_directories(${LIBRARY} PRIVATE ${AMX_PATH})
            endif()

            if(CMAKE_UNITY_BUILD)
                set_target_properties(${LIBRARY} PROPERTIES UNITY_BUILD_MODE GROUP)
            endif()

            foreach(SOURCE ${SOURCES_UNITY} ${SOURCES_NO_UNITY} ${SOURCES_IMPL} ${SOURCES_${PARAMETER_SET}})
                set(SOURCE_FULL_PATH ${ALLOC}/neon-${PARAMETER_SET}/${SOURCE})
                target_sources(${LIBRARY} PRIVATE ${SOURCE_FULL_PATH})

                if(CMAKE_UNITY_BUILD AND NOT SOURCE IN_LIST SOURCES_NO_UNITY)
                    set_source_files_properties(${SOURCE_FULL_PATH} PROPERTIES UNITY_GROUP "${LIBRARY}")
                endif()
            endforeach()

            foreach(DUPLICATE_SYMBOL ${DUPLICATE_SYMBOLS})
                target_compile_definitions(${LIBRARY} PRIVATE ${DUPLICATE_SYMBOL}=${LIBRARY}_${DUPLICATE_SYMBOL})
            endforeach()

            target_include_directories(${LIBRARY} PUBLIC
                ${ALLOC}/neon-${PARAMETER_SET} ${HASH_PATH} ${SORT_PATH} ${RAND_PATH})

            if(NOT(PARAMETER_SET STREQUAL hps2048677 AND IMPL STREQUAL amx))
                list(APPEND SPEED_PREFIXES ${SPEED_POLYMUL_PREFIX})
            endif()

            foreach(SPEED_PREFIX SPEED_SOURCE SPEED_NTESTS IN ZIP_LISTS SPEED_PREFIXES SPEED_SOURCES SPEED_NTESTSS)
                set(SPEED ${SPEED_PREFIX}_${LIBRARY})

                add_executable_with_symlink(${SPEED} ${ALLOC}/neon-${PARAMETER_SET}/${SPEED_SOURCE})
                target_compile_definitions(${SPEED} PRIVATE NTESTS=${SPEED_NTESTS})

                if(${SPEED_PREFIX} STREQUAL speed_polymul_ct)
                    target_compile_definitions(${SPEED} PRIVATE ITERS=1000000)
                endif()

                target_link_libraries(${SPEED} PRIVATE ${LIBRARY} neon_rng cycles)
            endforeach()

            add_executable_with_symlink(${PQCGENKAT_KEM}
                ${CMAKE_SOURCE_DIR}/reference/Reference_Implementation/crypto_kem/ntru${PARAMETER_SET}/PQCgenKAT_kem.c)
            target_compile_options(${PQCGENKAT_KEM} PRIVATE -Wno-unused-result)
            target_link_libraries(${PQCGENKAT_KEM} PRIVATE ${LIBRARY})

            ADD_KAT_TESTS()

            if(NOT PARAMETER_SET STREQUAL hrss701)
                target_sources(${LIBRARY} PRIVATE ${SORT_SOURCES})
            endif()
        endforeach()
    endforeach()
endforeach()
