#ifndef PROBLEM_DATA_HEADER
#define PROBLEM_DATA_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "fused_constraint.h"

template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class ProblemData {
private:

    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;

    /* These are the derivatives. The i^th column holds the (i+1)^th derivatives */
    Eigen::Matrix<T, Get::n_vars, Eigen::Dynamic> derivatives;

public:

    const Eigen::Matrix<T, n_x, n_w> waypoints;
    const Eigen::Matrix<T, n_x, 1> initial_state;
    const Get getter;
    FusedConstraint<T, Index, Bool, Data, Get::n_vars> fused_constraint;

    ProblemData(const Eigen::Matrix<T, n_x, n_w> &waypoints,
                const Eigen::Matrix<T, n_x, 1> &initial_state,
                const Get &getter,
                const FusedConstraint<T, Index, Bool, Data, Get::n_vars> &fused_constraint)
            : waypoints(waypoints),
              initial_state(initial_state),
              getter(getter),
              fused_constraint(fused_constraint) {
        derivatives = Eigen::Matrix<T, Get::n_vars, Eigen::Dynamic>::Zero(Get::n_vars, 0);
    }


    /**
     * Recompute all of the derivatives up to the order specified by "upToDerivative".
     * For example, if upToDerivative = 3, then we compute the 1st, 2nd, and 3rd derivatives.
     * To get these values, call the getDerivatives method.
     */
    void recomputeDerivatives(T *x0, Index upToDerivative) {

        /* It doesn't make sense to call this function with upToDerivative < 1 */
        assert(upToDerivative > 0);

        /* Make sure that we have allocated enough space to hold the derivatives */
        if (derivatives.cols() < upToDerivative) {
            derivatives.resize(Get::n_vars, upToDerivative);
        }

        /* Now compute the derivatives */
        T *x = x0;
        for (Index i = 0; i < upToDerivative; ++i) {
            T *dx = derivatives.col(i).data();
            this->getter.derivatives(x, dx);
            x = dx;
        }
    }

    /**
     * Recompute all of the unscaled derivatives.
     */
    void recomputeUnscaledDerivatives(T *x0, Index upToDerivative) {

        /* It doesn't make sense to call this function with upToDerivative < 1 */
        assert(upToDerivative > 0);

        /* Make sure that we have allocated enough space to hold the derivatives */
        if (derivatives.cols() < upToDerivative) {
            derivatives.resize(Get::n_vars, upToDerivative);
        }

        /* Now compute the derivatives */
        T *x = x0;
        for (Index i = 0; i < upToDerivative; ++i) {
            T *dx = derivatives.col(i).data();
            this->getter.derivativesUnscaled(x, dx);
            x = dx;
        }
    }

    /** Return the specified derivative of the data from the last time that you called
     * recomputeDerivatives. For instance, specify derivative=1 means that this function
     * will return the first derivative, etc. Note that if you never called recomputeDerivatives
     * or if you never called it with a degree >= the order that you specify here, then
     * an error is thrown.
     */
    T *getDerivatives(Index derivative) {
        assert(derivative <= derivatives.cols());
        return derivatives.col(derivative - 1).data();
    }
};

#endif /* PROBLEM_DATA_HEADER */