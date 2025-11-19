cmake_minimum_required(VERSION 3.18)
project(FANL_DLL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


option(WITH_OPENCV "Enable OpenCV for image loading" OFF)

add_executable(fanl_test fanl_test.cpp)
target_link_libraries(fanl_test "${TORCH_LIBRARIES}")
set_property(TARGET fanl_test PROPERTY CXX_STANDARD 17)


if(WITH_OPENCV)
  find_package(OpenCV REQUIRED)
  target_compile_definitions(fanl_test PRIVATE USE_OPENCV=1)
  target_link_libraries(fanl_test ${OpenCV_LIBS})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fanl_dll.cpp")
  add_library(fanl_dll SHARED fanl_dll.cpp)
  target_link_libraries(fanl_dll "${TORCH_LIBRARIES}")
  set_property(TARGET fanl_dll PROPERTY CXX_STANDARD 17)
endif()


set_target_properties(fanl_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
