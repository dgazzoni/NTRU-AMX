macro(create_symlink src dest target)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}
        DEPENDS ${dest})
endmacro()

function(add_executable_with_symlink executable)
    add_executable(${executable} ${ARGN})

    if(NOT ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_BINARY_DIR})
        create_symlink(
            ${CMAKE_CURRENT_BINARY_DIR}/${executable} ${CMAKE_BINARY_DIR}/${executable} ${executable})
        set_target_properties(${executable} PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_BINARY_DIR}/${executable})
    endif()
endfunction()
