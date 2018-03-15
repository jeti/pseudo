#ifndef SMOOTH_CONTROL_CONSTRAINTS_HEADER
#define SMOOTH_CONTROL_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"
#include "problem_data.h"

/**
 * This set of constraints simply ensures that the first derivatives
 * of the controls are equal at the overlapping collocation points.
 *
 * For example, the points waypoint i, collocation 0 and waypoint i-1, collocation point n_c
 * occur at the same time. However, we can produce estimates of the derivatives of the controls
 * at these points from the left and the right, that is, we can compute u_dot at that time
 * by using all of the control values in waypoint i (collocation points 0, ... n_c) amd we can
 * also produce an estimate of u_dot at that time by using all of the control values corresponding
 * to waypoint i-1 (collocation points 0,...n_c). We want to ensure that the control is smooth,
 * so we enforce the fact that these estimates are equal.
 *
 * This gives us n_u * (n_w-1) constraints.
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class SmoothControlConstraints
        : public EqualityConstraint<T, Index, Bool, Data, VariableGetter<T, Index, n_x, n_u, n_c, n_w>::n_vars> {
private:

    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
    using GetIndex = VariableGetter<Index, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<T, n_u, n_w - 1 >>;
    using PD = ProblemData<T, Index, Bool, Data, n_x, n_u, n_c, n_w>;

    static const Index n_constraints = n_u * (n_w - 1);

    Eigen::Matrix<T, n_c, 1> c_end_multipliers;
    Eigen::Matrix<T, n_c, 1> c_0_multipliers;

public:

    ~SmoothControlConstraints() {};

    Index getNumberOfConstraints() const override {
        return n_constraints;
    }

    Index evaluate(T *x,
                   Bool new_x,
                   T *g,
                   Data *data,
                   Index derivatives_in_data = 0) override {

        /* Cast the pointer to a ProblemData pointer */
        PD *pd = static_cast<PD *>(data);

        /* We need the first derivatives. If they aren't provided, then we need to calculate them */
        if (derivatives_in_data < 1) {
            derivatives_in_data = 1;
            pd->recomputeDerivatives(x, 1);
        }

        /* Get all of the control derivatives at the first and last collocation points */
        T *dx = pd->getDerivatives(1);
        auto c_0 = Get::controlsAtCollocationPoint(dx, 0);
        auto c_end = Get::controlsAtCollocationPoint(dx, n_c - 1);

        /* We are going to compute c_0 (of waypoint i) - c_end (of waypoint i-1)
         * This will be a matrix of size n_u x (n_w-1) */
        Map G(g);
        G = c_0.template rightCols<n_w - 1>() - c_end.template leftCols<n_w - 1>();
        return derivatives_in_data;
    }

    Index evaluateJacobian(T *x,
                           Bool new_x,
                           T *dgdx,
                           Data *data,
                           Index derivatives_in_data = 0) override {

        /* Cast the pointer to a ProblemData pointer */
        PD *pd = static_cast<PD *>(data);

        /* We need the unscaled first derivatives. If they aren't provided, then we need to calculate them */
        if (derivatives_in_data < 1) {
            derivatives_in_data = 1;
            pd->recomputeUnscaledDerivatives(x, 1);
        }
        T *dx = pd->getDerivatives(1);

        /* For each waypoint ... */
        Eigen::Ref<Eigen::Matrix<T, 1, n_w>> times = Get::times(x);
        for (Index i_w = 0; i_w < n_w; ++i_w) {

            T t_w = times(i_w);
            if (i_w < n_w - 1) {

                /* First, put in the terms associated with the interpolation polynomial.
                 * This involves copying the derivative coefficients n_u times for each waypoint
                 * and multiplying by the time taken to reach that waypoint. */
                Eigen::Map<Eigen::Matrix<T, n_c * n_u, 1>> dGdX(dgdx);
                dGdX = c_end_multipliers.template replicate<n_u, 1>().array() / t_w;
                dgdx += n_c * n_u;

                /* Now the jacobian components with respect to the times */
                Eigen::Map<Eigen::Matrix<T, n_u, 1>> dGdTime(dgdx);
                dGdTime = Get::control(dx, n_c - 1, i_w) / (t_w * t_w);
                dgdx += n_u;
            }
            if (i_w > 0) {

                Eigen::Map<Eigen::Matrix<T, n_c * n_u, 1>> dGdX(dgdx);
                dGdX = c_0_multipliers.template replicate<n_u, 1>().array() / t_w;
                dgdx += n_c * n_u;

                Eigen::Map<Eigen::Matrix<T, n_u, 1>> dGdTime(dgdx);
                dGdTime = -Get::control(dx, 0, i_w) / (t_w * t_w);
                dgdx += n_u;
            }
        }
        return derivatives_in_data;
    }

    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) override {

        /* We need the right derivative coefficients so that we can efficiently construct the jacobian */
        PD *pd = static_cast<PD *>(data);
        Eigen::Matrix<T, n_c, n_c> derivative_coefficients = pd->getter.rightDerivativeCoefficients();
        c_0_multipliers = derivative_coefficients.col(0);
        c_end_multipliers = -derivative_coefficients.col(n_c - 1);

        /* Make sure that the number of nonzero rows and columns are properly sized */
        this->nonzero_rows.resize(2 * (n_c + 1) * n_constraints);
        this->nonzero_cols.resize(2 * (n_c + 1) * n_constraints);
        Index *rows = this->nonzero_rows.data();
        Index *cols = this->nonzero_cols.data();

        /* To get the nonzero columns, we essentially replace
         * x with a list of indices in the constraint function, and then
         * see which values are pulled off. */
        Eigen::Matrix<Index, Get::n_vars, 1> inds;
        for (Index i = 0; i < Get::n_vars; ++i)
            inds[i] = i;

        /* Create a list of rows that looks like
         * 0, ..., 0, 1, ..., 1, */
        Eigen::Matrix<Index, n_c * n_u, 1> count;
        for (Index i_u = 0; i_u < n_u; ++i_u)
            count.template middleRows<n_c>(i_u * n_c).fill(i_u);

        /* Create a list that looks like 0, ... , n_u-1 */
        Eigen::Matrix<Index, n_u, 1> count_n_u;
        for (Index i_u = 0; i_u < n_u; ++i_u)
            count_n_u(i_u) = i_u;

        for (Index i_w = 0; i_w < n_w; ++i_w) {

            if (i_w < n_w - 1) {

                /* First, put in the terms associated with the interpolation polynomial.
                 * This involves copying the derivative coefficients n_u times for each waypoint
                 * and multiplying by the time taken to reach that waypoint. */
                Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>> Rows(rows);
                Rows.array() = count.array() + i_w * n_u;
                rows += n_c * n_u;

                Eigen::Map<Eigen::Matrix<Index, n_c, n_u>> Cols(cols);
                Cols = GetIndex::controlsAtWaypoint(inds.data(), i_w).transpose();
                cols += n_c * n_u;

                /* Now the jacobian components with respect to the times */
                Eigen::Map<Eigen::Matrix<Index, n_u, 1>> timeRows(rows);
                timeRows.array() = count_n_u.array() + i_w * n_u;
                rows += n_u;

                Eigen::Map<Eigen::Matrix<Index, n_u, 1>> timeCols(cols);
                timeCols.fill(GetIndex::times(inds.data())(i_w));
                cols += n_u;
            }
            if (i_w > 0) {

                Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>> Rows(rows);
                Rows.array() = count.array() + (i_w - 1) * n_u;
                rows += n_c * n_u;

                Eigen::Map<Eigen::Matrix<Index, n_c, n_u>> Cols(cols);
                Cols = GetIndex::controlsAtWaypoint(inds.data(), i_w).transpose();
                cols += n_c * n_u;

                Eigen::Map<Eigen::Matrix<Index, n_u, 1>> timeRows(rows);
                timeRows.array() = count_n_u.array() + (i_w - 1) * n_u;
                rows += n_u;

                Eigen::Map<Eigen::Matrix<Index, n_u, 1>> timeCols(cols);
                timeCols.fill(GetIndex::times(inds.data())(i_w));
                cols += n_u;
            }
        }
    }
};

#endif /* SMOOTH_CONTROL_CONSTRAINTS_HEADER */