#include "solver_qpOASES.h"
#include <iostream>

SolverQPOASES::SolverQPOASES()
    : first_iteration_(true), sqp_(nullptr)
{
}

SolverQPOASES::~SolverQPOASES(void)
{
}

bool SolverQPOASES::solve(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                          const Eigen::MatrixXd &A, const Eigen::VectorXd &lb, 
                          const Eigen::VectorXd &ub, const Eigen::VectorXd &lbA, 
                          const Eigen::VectorXd &ubA)
{
    int num_variables = g.size();   // n_v (number of variables)
    int num_constraints = A.rows(); // n_c (number of constraints)

    if (first_iteration_)
    {
        sqp_ = std::unique_ptr<qpOASES::SQProblem>(new qpOASES::SQProblem(num_variables, num_constraints));
    }

    auto options = sqp_->getOptions();
    // options.printLevel = qpOASES::PL_NONE;
    if (VERBOSE)
    {
        options.printLevel = qpOASES::PL_LOW;
    }
    // options.enableFarBounds = qpOASES::BT_TRUE;
    // options.enableFlippingBounds = qpOASES::BT_TRUE;
    options.enableRamping = qpOASES::BT_FALSE;
    options.enableNZCTests = qpOASES::BT_FALSE;
    // options.enableRegularisation = qpOASES::BT_TRUE;
    options.enableDriftCorrection = 0;
    options.terminationTolerance = 1e-6;
    options.boundTolerance = 1e-4;
    options.epsIterRef = 1e-6;
    sqp_->setOptions(options);

    // qpOASES uses row-major storing
    qpOASES::real_t *H_qp = new qpOASES::real_t[num_variables * num_variables];
    qpOASES::real_t *A_qp = new qpOASES::real_t[num_constraints * num_variables];
    qpOASES::real_t *g_qp = new qpOASES::real_t[num_variables];
    qpOASES::real_t *lb_qp = new qpOASES::real_t[num_variables];
    qpOASES::real_t *ub_qp = new qpOASES::real_t[num_variables];
    qpOASES::real_t *lbA_qp = new qpOASES::real_t[num_constraints];
    qpOASES::real_t *ubA_qp = new qpOASES::real_t[num_constraints];

    // conversion from Eigen to qpOASES_data
    convert_Eigen_mat_to_qpOASES_data(H_qp, H);
    convert_Eigen_mat_to_qpOASES_data(A_qp, A);
    convert_Eigen_vec_to_qpOASES_data(g_qp, g);
    convert_Eigen_vec_to_qpOASES_data(lb_qp, lb);
    convert_Eigen_vec_to_qpOASES_data(ub_qp, ub);
    convert_Eigen_vec_to_qpOASES_data(lbA_qp, lbA);
    convert_Eigen_vec_to_qpOASES_data(ubA_qp, ubA);

    qpOASES::SymSparseMat H_mat(H.rows(), H.cols(), H.cols(), H_qp);
    H_mat.createDiagInfo();
    qpOASES::SparseMatrix A_mat(A.rows(), A.cols(), A.cols(), A_qp);

    qpOASES::returnValue ret = qpOASES::TERMINAL_LIST_ELEMENT;
    max_iteration_ = 200;
    
    if (first_iteration_)
    {
        ret = sqp_->init(&H_mat, g_qp, &A_mat, lb_qp, ub_qp, lbA_qp, ubA_qp, max_iteration_);

        std::cout << "H: " << std::endl << H << std::endl;
        std::cout << "g: " << std::endl << g.transpose() << std::endl;
        std::cout << "A: " << std::endl << A << std::endl;
        std::cout << "lbA: " << std::endl << lbA.transpose() << std::endl;
        std::cout << "ubA: " << std::endl << ubA.transpose() << std::endl;
        std::cout << "lb: " << std::endl << lb.transpose() << std::endl;
        std::cout << "ub: " << std::endl << ub.transpose() << std::endl;
        first_iteration_ = false;
    }
    else
    {
        ret = sqp_->hotstart(&H_mat, g_qp, &A_mat, lb_qp, ub_qp, lbA_qp, ubA_qp, max_iteration_, &max_time_);
    }

    optimal_solution_ = Eigen::VectorXd::Zero(num_variables);
    sqp_->getPrimalSolution(optimal_solution_.data());
    // std::cout << "optimal solution: " << optimal_solution_.transpose() << std::endl;

    if (ret != qpOASES::SUCCESSFUL_RETURN)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void SolverQPOASES::convert_Eigen_mat_to_qpOASES_data(qpOASES::real_t *out, const Eigen::MatrixXd &in)
{
    for (int i = 0; i < in.rows(); i++)
    {
        for (int j = 0; j < in.cols(); j++)
        {
            out[i * in.cols() + j] = in(i, j);
        }
    }
}
void SolverQPOASES::convert_Eigen_vec_to_qpOASES_data(qpOASES::real_t *out, const Eigen::VectorXd &in)
{
    for (int i = 0; i < in.size(); i++)
    {
        out[i] = in(i);
    }
}