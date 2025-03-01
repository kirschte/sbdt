################################################################
# FOLDER STRUCTURE
################################################################

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)


################################################################
# libsbdt
################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libsbdt.cmake)


################################################################
# Example Application
################################################################

option(BUILD_EXAMPLE_APPLICATION "If set, CMake will add the c++ example application to the build targets." OFF)
if (BUILD_EXAMPLE_APPLICATION)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_example.cmake)
endif()