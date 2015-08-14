cmake_minimum_required (VERSION 2.8)
project (cligrid)
 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Lua REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
find_package(ITK REQUIRED)

#FIND_PACKAGE(Boost COMPONENTS thread system filesystem regex REQUIRED)
#IF (Boost_FOUND)
#    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#    ADD_DEFINITIONS( "-DHAS_BOOST" )
#ENDIF()

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
)
include(${ITK_USE_FILE})
 
set (SOURCES
    cli_grid.cpp
)

add_library(itktotensor itktotensor)
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries (${PROJECT_NAME} ${ITK_LIBRARIES}
					   ${LUA_LIBRARIES} luaT TH itktotensor)# luaT

