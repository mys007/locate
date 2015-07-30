cmake_minimum_required (VERSION 2.8)
project (luawrap)
 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Lua REQUIRED)
FIND_PACKAGE(Torch REQUIRED)

include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
)
 
set (SOURCES
    main.cpp
)
 
add_executable (${PROJECT_NAME} ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${LUA_LIBRARIES} luaT TH)

