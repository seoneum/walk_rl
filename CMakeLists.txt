cmake_minimum_required(VERSION 3.16)
project(walk_rl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_VIEWER      "Build viewer (needs GLFW)" ON)
option(BUILD_ROS2_NODE   "Build ROS2 inference node" OFF)
option(USE_OPENMP        "Enable OpenMP" ON)
option(USE_MUJOCO_CONFIG "Use find_package(mujoco) if available" ON)

# LibTorch (견고한 탐색 + 호환용 타깃 생성)
# 1) 환경 변수에서 힌트 수집
if(NOT Torch_DIR AND DEFINED ENV{Torch_DIR})
  set(Torch_DIR $ENV{Torch_DIR})
endif()
if(DEFINED ENV{LIBTORCH_HOME})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_HOME}")
  if(NOT Torch_DIR)
    set(Torch_DIR "$ENV{LIBTORCH_HOME}/share/cmake/Torch")
  endif()
endif()

# 2) Torch 찾기
find_package(Torch REQUIRED CONFIG)

message(STATUS "Found LibTorch at: ${Torch_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 3) 호환용 임포트 타깃 생성 (가장 중요한 부분)
if(NOT TARGET Torch::Torch)
  message(STATUS "Target Torch::Torch not found, creating compatibility target.")
  add_library(Torch::Torch INTERFACE IMPORTED)
  target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
  set_property(TARGET Torch::Torch PROPERTY INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}")
endif()


# MuJoCo (package -> fallback)
if(USE_MUJOCO_CONFIG)
  find_package(mujoco QUIET)
endif()
if(NOT mujoco_FOUND)
  message(STATUS "mujoco package not found; fallback to manual search")
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
            HINTS $ENV{MUJOCO_DIR}/include /usr/include /usr/local/include)
  find_library(MUJOCO_LIB mujoco
               HINTS $ENV{MUJOCO_DIR}/lib $ENV{MUJOCO_DIR}/lib64 /usr/lib /usr/local/lib)
  if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIB)
    message(FATAL_ERROR "MuJoCo not found. Set MUJOCO_DIR or install mujoco.")
  endif()
  add_library(mujoco::mujoco UNKNOWN IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
  )
  message(STATUS "Using MuJoCo (manual): inc=${MUJOCO_INCLUDE_DIR}, lib=${MUJOCO_LIB}")
else()
  message(STATUS "Using MuJoCo (package): mujoco::mujoco")
endif()

# yaml-cpp (config)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP QUIET)
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  message(STATUS "OpenMP enabled")
endif()


# compile opts / RPATH
function(configure_compile_opts TARGET_NAME)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -O3)
  endif()
endfunction()

function(configure_rpath TARGET_NAME)
  if(UNIX AND NOT APPLE)
    get_filename_component(_torch_prefix "${Torch_DIR}/../../.." ABSOLUTE)
    set(_torch_lib_dir "${_torch_prefix}/lib")
    get_target_property(_mujoco_lib mujoco::mujoco IMPORTED_LOCATION_RELEASE)
    if(NOT _mujoco_lib)
      get_target_property(_mujoco_lib mujoco::mujoco IMPORTED_LOCATION)
    endif()
    get_filename_component(_mujoco_lib_dir "${_mujoco_lib}" DIRECTORY)
    set_target_properties(${TARGET_NAME} PROPERTIES
      BUILD_RPATH "${_torch_lib_dir}:${_mujoco_lib_dir}"
      INSTALL_RPATH "${_torch_lib_dir}:${_mujoco_lib_dir}"
      INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endfunction()

# rl_core
add_library(rl_core STATIC
  src/mj_env.cpp
  src/ppo.cpp
  src/ars.cpp
  src/rl_config.cpp
  src/elite_buffer.cpp
  src/gait_ref.cpp
)
target_include_directories(rl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rl_core PUBLIC Torch::Torch mujoco::mujoco Threads::Threads yaml-cpp)
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(rl_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(rl_core PUBLIC USE_OPENMP)
endif()
configure_compile_opts(rl_core)
configure_rpath(rl_core)

# train exe
add_executable(train_mjppo src/train.cpp)
target_link_libraries(train_mjppo PRIVATE rl_core)
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(train_mjppo PRIVATE OpenMP::OpenMP_CXX)
endif()
configure_compile_opts(train_mjppo)
configure_rpath(train_mjppo)


# ROS2 node (optional)
if(BUILD_ROS2_NODE)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(control_msgs REQUIRED)
  add_executable(ros2_infer_node src/ros2_infer_node.cpp src/ppo.cpp)
  target_include_directories(ros2_infer_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ros2_infer_node PRIVATE Torch::Torch)
  ament_target_dependencies(ros2_infer_node rclcpp std_msgs sensor_msgs control_msgs)
  install(TARGETS ros2_infer_node DESTINATION lib/${PROJECT_NAME})
  configure_compile_opts(ros2_infer_node)
  configure_rpath(ros2_infer_node)
endif()
# GLFW (optional viewer)
if(BUILD_VIEWER)
    find_package(glfw3 QUIET)
    if(NOT glfw3_FOUND)
        message(WARNING "GLFW3 not found; viewer target disabled")
    set(BUILD_VIEWER ON)
    else()
        message(STATUS "Found GLFW3")
    endif()
endif()

# viewer target (only once)
if(BUILD_VIEWER AND NOT TARGET quad_viewer)
    add_executable(quad_viewer src/viewer.cpp)
    
    # rl_core가 PUBLIC으로 yaml-cpp를 링크하므로, 굳이 yaml-cpp를 다시 적을 필요는 없음.
    # 필요하면 명시적으로 추가해도 무방.
    target_link_libraries(quad_viewer PRIVATE rl_core glfw)
    
    # 만약 시스템에서 타깃명이 glfw가 아닌 glfw3::glfw라면 다음으로 교체:
    # target_link_libraries(quad_viewer PRIVATE rl_core glfw3::glfw)
    
    configure_compile_opts(quad_viewer)
    configure_rpath(quad_viewer)
endif()


message(STATUS "==================================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "LibTorch:          ${Torch_DIR}")
if(mujoco_FOUND)
  message(STATUS "MuJoCo:            package")
else()
  message(STATUS "MuJoCo:            manual")
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  message(STATUS "OpenMP:            Enabled")
else()
  message(STATUS "OpenMP:            Disabled")
endif()
message(STATUS "Build ROS2 node:   ${BUILD_ROS2_NODE}")
message(STATUS "Build viewer:      ${BUILD_VIEWER}")
message(STATUS "==================================================")
