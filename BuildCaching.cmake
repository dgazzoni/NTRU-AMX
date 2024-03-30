option(USE_BUILD_CACHING "use ccache or sccache to speed up compilation" ON)

if(USE_BUILD_CACHING)
    find_program(CCACHE_PATH ccache)

    if(CCACHE_PATH)
        message(STATUS "Using ccache")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PATH}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PATH}")
    else()
        find_program(SCCACHE_PATH sccache)

        if(SCCACHE_PATH)
            message(STATUS "Using sccache")
            set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PATH}")
            set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PATH}")
        else()
            message(STATUS "*** WARNING *** neither ccache nor sccache were found, "
                "please install either one to speed up compilation time")
        endif()
    endif()
endif()
