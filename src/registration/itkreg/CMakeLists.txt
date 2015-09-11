cmake_minimum_required (VERSION 2.8)

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)
project (EvalRigidRegistration)
 
SET(MEDIPATCH_DIR $ENV{HOME}/Multimodal/medipatch/)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Lua REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
find_package(ITK REQUIRED)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
    ${MEDIPATCH_DIR}/src/external/
    ${MEDIPATCH_DIR}src/registration/imgsimilarity/

    
)

include(${ITK_USE_FILE})
 
set (SimilaritySources
    #${MEDIPATCH_DIR}src/registration/imgsimilarity/SimilarityMetric.cpp
    #${MEDIPATCH_DIR}src/registration/imgsimilarity/NnSimilarityMetric.cpp
    #${MEDIPATCH_DIR}src/registration/imgsimilarity/MutualInfoSimilarityMetric.cpp
)


#add_library(SimilarityLibrary ${SimilaritySources})

add_executable(${PROJECT_NAME} EvalRigidRegistration.cpp)

target_link_libraries (${PROJECT_NAME} ${ITK_LIBRARIES}
			${LUA_LIBRARIES} luaT TH)

