/**
 * @file loss_function.h
 * @author James Svacha (jbsvacha@gmail.com)
 * @brief Abstract base class for a loss function
 * @version 0.1
 * @date 2020-07-13
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef LOSS_FUNCTION_
#define LOSS_FUNCTION_

#include <Eigen/Geometry>

namespace robust_least_squares
{

class LossFunction
{
  public:

    /**
     * @brief Construct a new Loss Function object
     * 
     */
    LossFunction();

    /**
     * @brief Evaluate the loss function
     * 
     * @param errors 
     * @return double 
     */
    virtual double evaluateFunction(const Eigen::VectorXd &errors) const = 0;

    /**
     * @brief Evaluate the gradient
     * 
     * @param errors 
     * @return Eigen::VectorXd 
     */
    virtual Eigen::VectorXd evaluateGradient(const Eigen::VectorXd &errors) const = 0;

    /**
     * @brief Evaluate the Hessian
     * 
     * @param errors 
     * @return Eigen::VectorXd 
     */
    virtual Eigen::MatrixXd evaluateHessian(const Eigen::VectorXd &errors) const = 0;
};

}

#endif