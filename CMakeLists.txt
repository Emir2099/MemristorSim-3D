cmake_minimum_required(VERSION 3.20)
project(MemristorSim3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG master
)
FetchContent_MakeAvailable(implot)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

add_library(imgui_backend
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_backend PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_compile_definitions(imgui_backend PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_compile_definitions(imgui_backend PUBLIC IMGUI_ENABLE_DOCKING)
target_link_libraries(imgui_backend PUBLIC glfw glad)

add_library(implot_lib
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR} ${imgui_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_backend)

file(GLOB SRC_FILES CONFIGURE_DEPENDS src/**/*.cpp)

add_executable(MemristorSim src/main.cpp ${SRC_FILES})
target_include_directories(MemristorSim PUBLIC src ${glm_SOURCE_DIR})
target_link_libraries(MemristorSim PRIVATE glfw glad imgui_backend implot_lib)
target_compile_definitions(MemristorSim PRIVATE IMGUI_ENABLE_DOCKING)

if(WIN32)
  target_link_libraries(MemristorSim PRIVATE opengl32)
endif()
