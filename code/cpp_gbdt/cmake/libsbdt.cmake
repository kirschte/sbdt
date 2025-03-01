include_guard(GLOBAL)


################################################################
# TARGET: libsbdt (make fast_shared)
################################################################

set(TARGET sbdt)
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
    ${SOURCE_DIR}/*.cpp
    ${SOURCE_DIR}/gbdt/*.cpp
)

file(GLOB PUBLIC_HEADER_FILES CONFIGURE_DEPENDS
    ${INCLUDE_DIR}/*.h
    ${INCLUDE_DIR}/gbdt/*.h
)

add_library(${TARGET} SHARED ${SOURCE_FILES})

target_include_directories(${TARGET}
    PUBLIC $<BUILD_INTERFACE:${INCLUDE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${INCLUDE_DIR}/gbdt>
    $<INSTALL_INTERFACE:include>
)


#################################################################
# LIBRARY LINKING
################################################################

find_library(LIB_SVML
    NAME libsvml.so libsvml.a
    HINTS /usr/local/lib /usr/lib
)

find_library(LIB_FMT
    NAME libfmt.so libfmt.a
    HINTS /usr/local/lib /usr/lib
)

target_link_libraries(${TARGET}
    PUBLIC ${LIB_SVML}
    PUBLIC ${LIB_FMT}
)


################################################################
# libsbdt BUILD FLAGS
################################################################

target_compile_options(${TARGET} PRIVATE
    -c 
    -Wall 
    -Wextra 
    -std=gnu++11 
    -pipe 
    -Ofast 
    -ffast-math 
    -march=native 
    -mveclibabi=svml 
    -pthread 
    -flto 
    -fPIC
)

target_link_options(${TARGET} PRIVATE
    -Wl,--no-as-needed 
    -pthread 
)

target_compile_definitions(${TARGET} PRIVATE
    # DDELTA=5e-8 
    # DMAX_ALPHA=1000
)


################################################################
# INSTALLING libsbdt
################################################################

set(CMAKE_CONFIG_FILE sbdt-config.cmake)

install(TARGETS ${TARGET}
    EXPORT ${TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
) 

install(FILES ${PUBLIC_HEADER_FILES} DESTINATION include)

install(DIRECTORY ${INCLUDE_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING
        PATTERN "*.h"
)


################################################################
# INSTALLING CMAKE PACKAGE
################################################################

set(CMAKE_NAMESPACE sbdt::)

export(TARGETS
    ${TARGET}
    NAMESPACE ${CMAKE_NAMESPACE}
    FILE ${CMAKE_CONFIG_FILE}
)

install(EXPORT ${TARGET}
    FILE ${CMAKE_CONFIG_FILE}
    NAMESPACE ${CMAKE_NAMESPACE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/sbdt
)
