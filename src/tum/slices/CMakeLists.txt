cmake_minimum_required(VERSION 2.8)
project(ExtractSlices)



find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

FIND_PACKAGE(Boost COMPONENTS thread system filesystem regex REQUIRED)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto ")


add_executable(ExtractSlices  extract_slices.cpp )
target_link_libraries(ExtractSlices ${ITK_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY}
${Boost_SYSTEM_LIBRARY}
${Boost_REGEX_LIBRARY} )
