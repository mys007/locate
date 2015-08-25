cmake_minimum_required (VERSION 2.8)
project (SimilarityMetric)
 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Lua REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
find_package(ITK REQUIRED)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
)
include(${ITK_USE_FILE})
 
set (SOURCES
    main.cpp
)

#
add_library(Similarity SimilarityMetric.cpp NnSimilarityMetric.cpp  MutualInfoSimilarityMetric.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries (Similarity ${ITK_LIBRARIES}
		       ${LUA_LIBRARIES} luaT TH Similarity)
#target_link_libraries (${PROJECT_NAME} ${ITK_LIBRARIES}
#		       ${LUA_LIBRARIES} luaT TH Similarity)# luaT
target_link_libraries (${PROJECT_NAME} Similarity)

