include_guard(GLOBAL)


################################################################
# LINKS TO DATASETS
################################################################

# links to download the datasets from
set(DOWNLOAD_LINKS
    https://archive.ics.uci.edu/static/public/1/abalone.zip
    https://archive.ics.uci.edu/static/public/2/adult.zip
    https://archive.ics.uci.edu/static/public/94/spambase.zip
)


################################################################
# FOLDER CONFIGURATIONS (do not change)
################################################################

unset(TEST_DATASET_DIRECTORY CACHE) # remove from cache to prevent bugs when changing path without rebuild
if("${TEST_DATASET_DIRECTORY}" STREQUAL "")
    set(TEST_DATASET_DIRECTORY ${CMAKE_BINARY_DIR}/datasets) # defulat = <build>/datasets
elseif(NOT EXISTS ${TEST_DATASET_DIRECTORY})
    message(FATAL_ERROR
        "Specified TEST_DATASET_DIRECTORY '${TEST_DATASET_DIRECTORY}' does not exist so test datasets cannot be downloaded. "
        "Please create the directory or set TEST_DATASET_DIRECTORY to a valid path."
    )
endif()

# derived paths
set(DOWNLOAD_DIR ${TEST_DATASET_DIRECTORY}/zip) # path to download archives to
set(UNZIP_PATH ${TEST_DATASET_DIRECTORY}/unzip) # path to unzip archives to
set(REAL_PATH ${TEST_DATASET_DIRECTORY}/real) # path to copy just the datasets to

# guarantee paths exist before execution
file(MAKE_DIRECTORY ${DOWNLOAD_DIR})
file(MAKE_DIRECTORY ${UNZIP_PATH})
file(MAKE_DIRECTORY ${REAL_PATH})


################################################################
# DOWNLOAD and UNZIP
################################################################

foreach(LINK ${DOWNLOAD_LINKS})
    # Extract the filename (including ending) from each link
    # This is used to name the datasets accordingly
    set(LINK_SPLIT ${LINK}) # deep copy links
    string(REPLACE "/" ";" LINK_SPLIT ${LINK_SPLIT}) # split link at '/'
    list(GET LINK_SPLIT -1 ZIP_NAME) # take last element

    # Full Filename + Path of the to be downloaded ZIP archive
    set(ZIP_ARCHIVE_FILE ${DOWNLOAD_DIR}/${ZIP_NAME})

    # DOWNLOAD
    file(DOWNLOAD
        ${LINK}
        ${ZIP_ARCHIVE_FILE}
        SHOW_PROGRESS
    )

    # UNZIP FILE
    execute_process(
        WORKING_DIRECTORY ${UNZIP_PATH}
        COMMAND ${CMAKE_COMMAND} -E tar xfv ${ZIP_ARCHIVE_FILE}
    )
endforeach()


################################################################
# COPY all unzipped datasets
################################################################

# copy all datasets
file(GLOB DATA_FILES
    ${UNZIP_PATH}/*.data
)

file(COPY ${DATA_FILES} DESTINATION ${REAL_PATH})
