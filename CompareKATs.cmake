# https://stackoverflow.com/a/3071370/523079
macro(EXEC_CHECK CMD)
    execute_process(
        COMMAND ${CMAKE_BINARY_DIR}/${CMD}
        WORKING_DIRECTORY ${WORKING_DIRECTORY}
        RESULT_VARIABLE CMD_RESULT)

    if(CMD_RESULT)
        message(FATAL_ERROR "Error running ${CMD}: CMD_RESULT = ${CMD_RESULT}")
    endif()
endmacro()

macro(DIFF_FILES FILE1 FILE2)
    # https://stackoverflow.com/a/54354122/523079
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E compare_files ${FILE1} ${WORKING_DIRECTORY}/${FILE2}
        RESULT_VARIABLE compare_result)

    if(compare_result EQUAL 1)
        message(FATAL_ERROR "Error comparing KATs: KATs are different: ${FILE1} ${WORKING_DIRECTORY}/${FILE2}")
    elseif(NOT compare_result EQUAL 0)
        message(FATAL_ERROR "Error comparing KATs: unknown error: ${FILE1} ${WORKING_DIRECTORY}/${FILE2}")
    endif()
endmacro()

file(MAKE_DIRECTORY ${WORKING_DIRECTORY})
exec_check(${KATgen_cmd})
diff_files(${KAT_expected}.req ${KAT_actual}.req)
diff_files(${KAT_expected}.rsp ${KAT_actual}.rsp)
