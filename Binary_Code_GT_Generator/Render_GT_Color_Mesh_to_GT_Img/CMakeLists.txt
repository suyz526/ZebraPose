cmake_minimum_required(VERSION 3.0)
project(dataset_generator_coarse_to_fine)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

#set glad 
set(GLAD_DIR "render_related_source/glad")
add_library(glad SHARED "${GLAD_DIR}/src/glad.c")
include_directories("${GLAD_DIR}/include")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

###################
FIND_PACKAGE(OpenGL REQUIRED)
ADD_DEFINITIONS(${OpenGL_DEFINITIONS})
###################
find_package(Eigen3 3.3.4 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})
###################
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIR})
###################
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})
link_directories(${ASSIMP_LIBRARY_DIRS})
###################
find_package( OpenCV 3.0 REQUIRED )
###################

add_library(opengl_render SHARED "render_related_source/opengl_render.cpp")
target_link_libraries(opengl_render ${OPENGL_LIBRARIES} glad glfw ${ASSIMP_LIBRARIES} ${OpenCV_LIBS})
