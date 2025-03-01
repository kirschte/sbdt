include(${CMAKE_CURRENT_LIST_DIR}/libsbdt.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/download_datasets.cmake)


################################################################
# TARGET: run_example (make fast_shared)
################################################################

set(TARGET run_example)
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
    ${SOURCE_DIR}/*.cpp
)

add_executable(${TARGET} ${SOURCE_FILES})


################################################################
# LIBRARY LINKING
################################################################

target_link_libraries(${TARGET}
    PUBLIC sbdt
)


################################################################
# libsbdt BUILD FLAGS
################################################################

target_compile_options(${TARGET} PRIVATE
    -c
    #-Werror
    -std=gnu++11
    -Ofast
    -pipe
    -ffast-math
    -march=native
    -mveclibabi=svml
    -pthread
    -flto
    -fPIC
)
