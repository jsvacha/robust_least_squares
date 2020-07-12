/**
 * @file huber_loss_function.h
 * @author your name (you@domain.com)
 * @brief: Huber loss function for robust least squares
 * @version 0.1
 * @date 2020-07-12
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef HUBER_LOSS_FUNCTION_
#define HUBER_LOSS_FUNCTION_

#include <Eigen/Geometry>

namespace robust_least_squares
{

class HuberLossFunction
{
  public:
    /**
     * @brief Construct a new Huber Loss Function
     * 
     * @param x_transition: Domain value for which function transitions from quadratic to linear
     */
    HuberLossFunction(const double x_transition);

    /**
     * @brief Evaluate Huber loss function
     * 
     * @param errors: Vector of errors
     * @return double 
     */
    double evaluateFunction(const Eigen::VectorXd &errors) const;

    /**
     * @brief Evaluate Huber loss function gradient on a vector of errors
     * 
     * @param errors: Vector of errors
     * @return Eigen::VectorXd 
     */
    Eigen::VectorXd evaluateGradient(const Eigen::VectorXd &errors) const;

    /**
     * @brief Evaluate loss Hessian  on a vector of errors
     * 
     * @param errors: Vector of errors 
     * @return Eigen::MatrixXd 
     */
    Eigen::MatrixXd evaluateHessian(const Eigen::VectorXd &errors) const;

  private:
    double x_transition_;

};

}

#endif
