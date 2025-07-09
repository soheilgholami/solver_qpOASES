#ifndef _OPTIMIZATION_SOLVER_QPOASES_H_
#define _OPTIMIZATION_SOLVER_QPOASES_H_

#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include "qpOASES.hpp"

const bool VERBOSE = true;

USING_NAMESPACE_QPOASES


class SolverQPOASES
{
public:
    SolverQPOASES(void);
    ~SolverQPOASES(void);

    void set_max_time(const double in) { max_time_ = in; }
    void set_max_iteration(const int in) { max_iteration_ = in; }

    bool solve(const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::MatrixXd &,
               const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &, 
               const Eigen::VectorXd &);

    Eigen::VectorXd get_optimal_solution(void) { return optimal_solution_; }

protected:
    void convert_Eigen_mat_to_qpOASES_data(qpOASES::real_t *, const Eigen::MatrixXd &);
    void convert_Eigen_vec_to_qpOASES_data(qpOASES::real_t *, const Eigen::VectorXd &);

private:
    std::unique_ptr<qpOASES::SQProblem> sqp_; // SQProblem: variable matrices (sequential)

    int max_iteration_{1000};
    double max_time_{5.0};
    bool first_iteration_{true};

    Eigen::VectorXd optimal_solution_;
};

#endif