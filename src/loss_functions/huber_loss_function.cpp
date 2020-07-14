/**
 * @file huber_loss_function.cpp
 * @author James Svacha (jbsvacha@gmail.com)
 * @brief Implementation of Huber loss function
 * @version 0.1
 * @date 2020-07-12
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "robust_least_squares/loss_functions/huber_loss_function.h"
#include "math.h"

namespace robust_least_squares
{

int sign(const double input)
{
  return (input > 0) - (input < 0);
}

HuberLossFunction::HuberLossFunction(const double x_transition) : 
  LossFunction(), 
  x_transition_{x_transition}
{
}

double HuberLossFunction::evaluateFunction(const Eigen::VectorXd &errors) const
{
  double result = 0;
  for (uint i = 0; i < errors.size(); ++i)
  {
    if (std::abs(errors(i)) < x_transition_)
      result += errors(i) * errors(i);
    else
      result += x_transition_ * (2 * std::abs(errors(i)) - x_transition_);
  }
  return result;
}

Eigen::VectorXd HuberLossFunction::evaluateGradient(const Eigen::VectorXd &errors) const
{
  Eigen::VectorXd result = Eigen::VectorXd::Zero(errors.size());
  for (uint i = 0; i < errors.size(); ++i)
  {
    if (std::abs(errors(i)) < x_transition_)
      result(i) = 2 * errors(i);
    else
      result(i) = 2 * sign(errors(i)) * x_transition_;
  }
  return result;
}

Eigen::MatrixXd HuberLossFunction::evaluateHessian(const Eigen::VectorXd &errors) const
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(errors.size(), errors.size());
  for (uint i = 0; i < errors.size(); ++i)
  {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(errors.size());
    if (std::abs(errors(i)) < x_transition_)
      gradient(i) = 2.0;
    result.row(i) = gradient;
  }
  return result;
}

} // namespace robust_least_squares
