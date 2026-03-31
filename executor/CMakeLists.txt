cmake_minimum_required(VERSION 3.10)
project(panda_executor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release with debug info ───────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Franka REQUIRED)        # libfranka — sets Franka::Franka target
find_package(Eigen3 REQUIRED)        # header-only

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(panda_executor src/panda_executor.cpp)

target_include_directories(panda_executor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(panda_executor
    PRIVATE
        Franka::Franka
        Eigen3::Eigen
)

# ── Real-time scheduling hint ─────────────────────────────────────────────────
# Allow the binary to set SCHED_FIFO without root (set-uid is not needed if
# you add the user to the 'realtime' group and raise rlimits via /etc/security/
# limits.conf instead).  This is informational only — CMake cannot grant the
# capability automatically.
message(STATUS
    "NOTE: For real-time execution run:\n"
    "  sudo setcap cap_sys_nice+ep ${CMAKE_BINARY_DIR}/panda_executor\n"
    "OR add your user to the 'realtime' group and configure limits.conf."
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS panda_executor DESTINATION bin)
