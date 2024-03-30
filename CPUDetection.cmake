execute_process(
    COMMAND sysctl -n machdep.cpu.brand_string
    OUTPUT_VARIABLE SYSCTL_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "M[1-9]" CPU "${SYSCTL_OUTPUT}")

add_compile_definitions(CPU_${CPU})

message(STATUS "Detected CPU: ${CPU}")
