#ifndef CHECK_JACOBIAN_HEADER
#define CHECK_JACOBIAN_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "constraint.h"

/** Convert a constraint's sparse jacobian representation to a full matrix  */
template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
Eigen::Matrix<T, Eigen::Dynamic, n_vars> full_jacobian(Constraint<T, Index, Bool, Data, n_vars> &constraint,
                                                       T *x,
                                                       Bool new_x,
                                                       Data *data) {

    /* Allocate space for the output */
    Eigen::Matrix<T, Eigen::Dynamic, n_vars> jacobian(constraint.getNumberOfConstraints(), n_vars);
    jacobian.setZero();

    /* Get the sparse representation */
    /* NOTE!!!!!!! We are purposefully generating the sparsity structure with a different x
     * so that we can simulate real circumstances in which the variables change with time. */
    Eigen::Matrix<T, n_vars, 1> random_x = Eigen::Matrix<T, n_vars, 1>::Random();
//     T* at_x = x;
    T* at_x = random_x.data();
    constraint.generateSparsityStructure(at_x, new_x, data);

    Eigen::Matrix<T, Eigen::Dynamic, 1> dgdx(constraint.getNumberOfJacobianNonzeros());
    constraint.evaluateJacobian(x, new_x, dgdx.data(), data);
    using Indices = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    Indices rows = constraint.getNonzeroJacobianRows();
    Indices cols = constraint.getNonzeroJacobianCols();

    /* Now fill the full jacobian */
    for (Index i = 0; i < constraint.getNumberOfJacobianNonzeros(); ++i) {
        jacobian(rows(i), cols(i)) = dgdx(i);
    }
    return jacobian;
}

/** Compute the jacobian of a constraint using finite differences */
template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
Eigen::Matrix<T, Eigen::Dynamic, n_vars> finite_difference(Constraint<T, Index, Bool, Data, n_vars> &constraint,
                                                           T *x,
                                                           Bool new_x,
                                                           Data *data,
                                                           const T dx = 1e-7) {

    /* Allocate space for the output */
    Eigen::Matrix<T, Eigen::Dynamic, n_vars> jacobian(constraint.getNumberOfConstraints(), n_vars);

    /* First evaluate the constraint at the initial point x */
    Eigen::Matrix<T, Eigen::Dynamic, 1> g0(constraint.getNumberOfConstraints());
    constraint.evaluate(x, new_x, g0.data(), data, 0);

    /* Next, evaluate the jacobian at the perturbed variables */
    for (Index i_var = 0; i_var < n_vars; i_var++) {

        const T x_var = x[i_var];
        x[i_var] += dx;
        constraint.evaluate(x, new_x, jacobian.col(i_var).data(), data, 0);
        x[i_var] = x_var;
    }
    jacobian.colwise() -= g0;
    jacobian /= dx;
    return jacobian;
}

/**
 * This method simply returns the difference between the finite difference jacobian
 * and a constraint's implementation.
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
Eigen::Matrix<T, Eigen::Dynamic, n_vars> check_jacobian(Constraint<T, Index, Bool, Data, n_vars> &constraint,
                                                        T *x,
                                                        Bool new_x,
                                                        Data *data) {
    auto full = full_jacobian(constraint, x, new_x, data);
    auto fd = finite_difference(constraint, x, new_x, data);
    return full - fd;
}

#endif /* CHECK_JACOBIAN_HEADER */