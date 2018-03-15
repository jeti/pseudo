#ifndef CONSTRAINT_HEADER
#define CONSTRAINT_HEADER

#include "Eigen/Dense"
#include <vector>

template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
class Constraint {
private:

    using Indices = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using Values = Eigen::Matrix<T, Eigen::Dynamic, 1>;

public:

    Indices nonzero_rows;
    Indices nonzero_cols;
    Values jacobian_values;

    virtual ~Constraint() {};

    /** Get the number of constraints */
    virtual Index getNumberOfConstraints() const = 0;

    /** A flag indicating whether this code should compute the offline jacobian values. */
    virtual Bool computeOfflineJacobianValues() const {
        return false;
    }

    /** Evaluate the constraint at x and store the values at in g.
     * Since it is quite common that a constraint will need to compute
     * the derivatives of x, those values should be placed in the data
     * pointer. Specifically, the derivatives_in_data variable tells
     * the constraint how many derivatives are precomputed in
     * the data pointer. For instance, if you need the second derivative
     * of x to compute the constraints, then either
     *
     * 1. You can call this function with derivatives_in_data = 0,
     * in which case, the constraint should assume that it has to
     * compute the derivatives itself.
     *
     * or
     *
     * 2. You can externally compute the derivatives and stash them in the
     * data pointer. In this case, you would pass derivatives_in_data = 2.
     *
     * Note that the return value of this function should indicate the
     * highest of level of derivatives in the data pointer after you are
     * finished evaluating this function. This is useful because computing
     * derivatives can be expensive, so if a constraint has to do it, then
     * it should stash the values in the data pointer so that another function
     * can use the values.
     */
    virtual Index evaluate(T *x,
                           Bool new_x,
                           T *g,
                           Data *data,
                           Index derivatives_in_data = 0) = 0;

    /** Evaluate the jacobian of the constraint at x and store the values in dgdx.
     * Specifically, you should only store the values listed in the locations returned by
     * getNonzeroJacobianRows and getNonzeroJacobianCols.
     *
     * For instance, let's suppose that the jacobian is a diagonal 3x3 matrix.
     * Then your getNonzeroJacobianRows and getNonzeroJacobianCols methods should both
     * return a vector {0,1,2}. This means that you should only be putting 3 values
     * into the vector dgdx... If the "full" 3x3 jacobian is called J, that would look like this
     *
     * dgdx[0] = J[0,0]
     * dgdx[1] = J[1,1]
     * dgdx[2] = J[2,2]
     *
     * As a slightly more complicated example, let's suppose that
     * the nonzero rows are [4,1,6,7] and
     * the nonzero cols are [5,2,8,9].
     *
     * Then this function should calculate J, and then set
     *
     * dgdx[0] = J[4,5]
     * dgdx[1] = J[1,2]
     * dgdx[2] = J[6,8]
     * dgdx[3] = J[7,9]
     *
     * The default implementation here assumes that the jacobian is independent of x.
     */
    virtual Index evaluateJacobian(T *x,
                                   Bool new_x,
                                   T *dgdx,
                                   Data *data,
                                   Index derivatives_in_data = 0) {
        Eigen::Map<Values> dGdX(dgdx, jacobian_values.size());
        dGdX = jacobian_values;
        return derivatives_in_data;
    }

    /** Evaluate the jacobian of the constraints at x,
     * and determine its sparsity structure.
     * Specifically, if g is an ( m x 1 ) vector, and x is ( n x 1 ),
     * then the jacobian will be ( m x n ).
     * This function should fill the vectors nonzero_rows and nonzero_cols
     * with the indices of nonzero entries in the jacobian.
     *
     * The default implementation here assumes that the jacobian is independent of x.
     */
    virtual void generateSparsityStructure(T *x,
                                           Bool new_x,
                                           Data *data) {

        /* To calculate the numerical jacobian, we first evaluate the constraint at the initial point x */
        Values g0(getNumberOfConstraints());
        evaluate(x, new_x, g0.data(), data, 0);

        /* Use finite differences to determine nonzero entries in the jacobian and their values.
         * Note that we are going to temporarily use STL containers for storing the number of
         * nonzero rows, cols, and values because they can be efficiently expanded with
         * push_back. Once we have those rows, columns, and values, we then copy them to Eigen.
         */
        Values g(getNumberOfConstraints());
        std::vector<T> jacobian_values__;
        std::vector<Index> nonzero_rows__;
        std::vector<Index> nonzero_cols__;
        const T dx = 1e-7;
        for (Index i_var = 0; i_var < n_vars; i_var++) {

            /* First we find the places where the jacobian will be nonzero by looking
             * for places which are Nan's. This indicates that the value in x affects
             * that location*/
            const T x_var = x[i_var];
            x[i_var] = NAN;
            evaluate(x, new_x, g.data(), data, 0);
            std::vector<T> tmp_rows__;
            for (Index i_constraint = 0; i_constraint < getNumberOfConstraints(); i_constraint++) {
                if (std::isnan(g[i_constraint])) {
                    tmp_rows__.push_back(i_constraint);
                    nonzero_rows__.push_back(i_constraint);
                    nonzero_cols__.push_back(i_var);
                }
            }

            /* Undo the perturbation for the next iteration, and rezero g */
            x[i_var] = x_var;
            g.setZero();

            /* Now we compute the actual values in the jacobian by looking at the values in the
             * above-mentioned entries. */
            if (computeOfflineJacobianValues()) {
                x[i_var] += dx;
                evaluate(x, new_x, g.data(), data);
                for (const auto &row : tmp_rows__) {
                    jacobian_values__.push_back((g[row] - g0[row]) / dx);
                }

                /* Undo the perturbation for the next iteration, and rezero g */
                x[i_var] = x_var;
                g.setZero();
            }
        }
        /* Now we will copy all of the values from the STL containers to Eigen */
        const Index n = jacobian_values__.size();
        jacobian_values.resize(n);
        nonzero_rows.resize(n);
        nonzero_cols.resize(n);
        for (Index i = 0; i < n; ++i) {
            jacobian_values(i) = jacobian_values__[i];
            nonzero_rows(i) = nonzero_rows__[i];
            nonzero_cols(i) = nonzero_cols__[i];
        }
    }

    /** Get the row indices of nonzero entries in the jacobian of the constraints */
    Indices getNonzeroJacobianRows() const {
        return nonzero_rows;
    }

    /** Get the column indices of nonzero entries in the jacobian of the constraints */
    Indices getNonzeroJacobianCols() const {
        return nonzero_cols;
    }

    /** Get the number of nonzero entries in the jacobian. */
    Index getNumberOfJacobianNonzeros() const {
        return nonzero_rows.size();
    }
};

#endif /* CONSTRAINT_HEADER */