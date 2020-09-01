/**
 * @file least_squares_optimizer.h
 * @author James Svacha (jbsvacha@gmail.com)
 * @brief: The robust least squares optmizer
 * @version 0.1
 * @date 2020-07-14
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef LEAST_SQUARES_OPTIMIZER_H_
#define LEAST_SQUARES_OPTIMIZER_H_

#include <robust_least_squares/loss_functions/huber_loss_function.h>
#include <memory>

namespace robust_least_squares 
{

enum class LOSS_FUNCTION_TYPE
{
  HUBER 
};

struct LeastSquaresOptimizerConfig
{
  LOSS_FUNCTION_TYPE type;
  double huber_transition;
  bool calculate_bias;
};

class LeastSquaresOptimizer
{
  public:
    /**
     * @brief Construct a new Least Squares Optimizer
     * 
     * @param config 
     */
    LeastSquaresOptimizer(const Eigen::MatrixXd &features, const Eigen::VectorXd &targets, const LeastSquaresOptimizerConfig &config);

    /**
     * @brief Get the Regression Solution
     * 
     * @return Eigen::VectorXd 
     */
    Eigen::VectorXd getRegression(void);

  private:
    LeastSquaresOptimizerConfig config_;
    std::shared_ptr<LossFunction> function_ptr_;
    Eigen::MatrixXd features_;
    Eigen::VectorXd targets_;
};

}

#endif