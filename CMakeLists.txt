cmake_minimum_required(VERSION 3.16.3)
project(robust_least_squares)

find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

add_library(librls
            src/loss_functions/loss_function.cpp
            src/loss_functions/huber_loss_function.cpp
)

add_executable(test_losses test/test_loss_functions.cpp)
target_link_libraries(test_losses Eigen3::Eigen librls)
