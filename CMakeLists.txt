cmake_minimum_required(VERSION 3.10.1)
project(meterReader)

set(project_name meterReader)
set(project_root ${PROJECT_SOURCE_DIR})
set(project_src ${project_root}/src)
set(project_include ${project_root}/include)
set(EXECUTABLE_OUTPUT_PATH ${project_root}/bin)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)
add_definitions(-pthread)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(OpenCV_DIR "/home/hit/Softwares/opencv-3.4.10/build")
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/hit/Softwares/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND! Please set ncnn_DIR environment variable")
else()
    message("ncnn found")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

include_directories(
    ${project_include}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(
    meterReader
    src/main.cpp
    src/yolov8_pose.cpp
    src/nanodet.cpp
)

target_link_libraries(
    meterReader
    ${OpenCV_LIBS}
    ncnn
)