cmake_minimum_required(VERSION 3.0.0)
project(mesh_segmentation)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

#################################
find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#################################
find_package( OpenCV 3.0 REQUIRED )


add_executable(Generate_Mesh_with_GT_Color Generate_Mesh_with_GT_Color.cpp)
target_link_libraries(Generate_Mesh_with_GT_Color ${PCL_LIBRARIES} ${OpenCV_LIBS})



