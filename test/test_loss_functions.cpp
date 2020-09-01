#include <robust_least_squares/loss_functions/huber_loss_function.h>
#include <iostream>

int main(int argc, char **argv)
{
  robust_least_squares::HuberLossFunction function(1.0);
  std::cout << "function: " << function.evaluateFunction(0.5*Eigen::Vector3d::Ones()) << std::endl;
  std::cout << "gradient: " << function.evaluateGradient(0.5*Eigen::Vector3d::Ones()).transpose() << std::endl;
  std::cout << "hessian: \n" << function.evaluateHessian(0.5*Eigen::Vector3d::Ones()) << std::endl;
  return 0;
}
