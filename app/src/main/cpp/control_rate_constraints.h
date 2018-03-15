#ifndef CONTROL_RATE_CONSTRAINTS_HEADER
#define CONTROL_RATE_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "inequality_constraint.h"
#include "problem_data.h"

/**
 * This set of constraints simply ensures that control rates lie within the specified bounds, that is,
 *
 * lower_bound <= u_dot <= upper_bound
 *
 * In general, an inequality bound is true if it is negative. So we reformulate the above to
 * the two bounds
 *
 * u_dot - upper_bound <= 0
 * lower_bound - u_dot <= 0
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class ControlRateConstraints
        : public InequalityConstraint<T,
                Index,
                Bool,
                Data,
                VariableGetter<T, Index, n_x, n_u, n_c, n_w>::n_vars> {
private:

    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
    using GetIndex = VariableGetter<Index, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<T, n_u, n_c>>;
    using PD = ProblemData<T, Index, Bool, Data, n_x, n_u, n_c, n_w>;

    static const Index n_constraints_2 = n_u * n_w * n_c;
    static const Index n_constraints = 2 * n_constraints_2;
    const Eigen::Matrix<T, n_u, 1> lower_bound;
    const Eigen::Matrix<T, n_u, 1> upper_bound;

    Eigen::Matrix<T, n_c * n_c, 1> derivative_coefficients;

public:

    ControlRateConstraints(const Eigen::Matrix<T, n_u, 1> &lower_bound,
                           const Eigen::Matrix<T, n_u, 1> &upper_bound)
            : lower_bound(lower_bound),
              upper_bound(upper_bound) {
    }

    ~ControlRateConstraints() {};

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

        /* First, we will put in all of the upper bound constraints */
        T *dx = pd->getDerivatives(1);
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            Map G(g);
            // TODO Use vector of bounds
            G = Get::controlsAtWaypoint(dx, i_w).colwise() - upper_bound;
            g += n_u * n_c;
        }

        /* Now the lower bounds */
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            Map G(g);
            // TODO Use vector of bounds
            G = (-Get::controlsAtWaypoint(dx, i_w)).colwise() + lower_bound;
            g += n_u * n_c;
        }
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
        T *negative_dgdx = dgdx + (n_c + 1) * n_constraints_2;
        for (Index i_w = 0; i_w < n_w; ++i_w) {

            /* First, put in the terms associated with the interpolation polynomial.
             * This involves copying the derivative coefficients n_u times for each waypoint
             * and multiplying by the time taken to reach that waypoint. */
            T t_w = times(i_w);
            Eigen::Map<Eigen::Matrix<T, n_c * n_c * n_u, 1>> dGdX(dgdx);
            dGdX = derivative_coefficients.template replicate<n_u, 1>().array() / t_w;
            dgdx += n_c * n_c * n_u;

            Eigen::Map<Eigen::Matrix<T, n_c * n_c * n_u, 1>> negative_dGdX(negative_dgdx);
            negative_dGdX = -dGdX;
            negative_dgdx += n_c * n_c * n_u;

            /* Now the jacobian components with respect to the times */
            Map G(dgdx);
            G = -Get::controlsAtWaypoint(dx, i_w) / (t_w * t_w);
            dgdx += n_c * n_u;

            Map G_negative(negative_dgdx);
            G_negative = -G;
            negative_dgdx += n_c * n_u;
        }
        return derivatives_in_data;
    }

    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) override {

        /* We need the right derivative coefficients so that we can efficiently construct the jacobian */
        PD *pd = static_cast<PD *>(data);
        Eigen::Matrix<T, n_c, n_c> dc = pd->getter.rightDerivativeCoefficients().transpose();
        derivative_coefficients = Eigen::Map<Eigen::Matrix<T, n_c * n_c, 1>>(dc.data());

        /* Make sure that the number of nonzero rows and columns are properly sized */
        this->nonzero_rows.resize((n_c + 1) * n_constraints);
        this->nonzero_cols.resize((n_c + 1) * n_constraints);
        Index *rows = this->nonzero_rows.data();
        Index *cols = this->nonzero_cols.data();

        const Index offset = (n_c + 1) * n_constraints_2;
        Index *negative_rows = rows + offset;
        Index *negative_cols = cols + offset;

        /* To get the nonzero columns, we essentially replace
         * x with a list of indices in the constraint function, and then
         * see which values are pulled off. */
        Eigen::Matrix<Index, Get::n_vars, 1> inds;
        for (Index i = 0; i < Get::n_vars; ++i)
            inds[i] = i;

        /* Let's do it! */
        Eigen::Matrix<Index, n_c * n_u, 1> count;
        for (Index i = 0; i < n_c * n_u; ++i)
            count(i) = i;

        Eigen::Matrix<Index, n_c, 1> count_u;
        for (Index i_c = 0; i_c < n_c; ++i_c)
            count_u(i_c) = i_c * n_u;
        Eigen::Matrix<Index, n_c * n_c, 1> row_count = count_u.template replicate<n_c, 1>();

        for (Index i_w = 0; i_w < n_w; ++i_w) {
            /* Set the rows. First we do the derivatives with respect to u. */
            for (Index i_u = 0; i_u < n_u; ++i_u) {
                Eigen::Map<Eigen::Matrix<Index, n_c * n_c, 1>>(rows).array() =
                        row_count.array() + (i_u + i_w * n_u * n_c);
                rows += n_c * n_c;

                Eigen::Map<Eigen::Matrix<Index, n_c * n_c, 1>>(negative_rows).array() =
                        row_count.array() + (i_u + i_w * n_u * n_c + n_constraints_2);
                negative_rows += n_c * n_c;
            }

            /* Next we do the derivatives with respect to time. */
            Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>>(rows).array() = count.array() + (i_w * n_u * n_c);
            rows += n_c * n_u;
            Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>>(negative_rows).array() =
                    count.array() + (i_w * n_u * n_c + n_constraints_2);
            negative_rows += n_c * n_u;

            /* Set the columns. First, we do the derivatives with respect to u */
            Eigen::Matrix<Index, n_c, n_u> u_indices = GetIndex::controlsAtWaypoint(inds.data(), i_w).transpose();
            Eigen::Map<Eigen::Matrix<Index, 1, n_u * n_c>> u_indices_vec(u_indices.data());
            Eigen::Map<Eigen::Matrix<Index, n_c, n_c * n_u>> columns(cols);
            Eigen::Map<Eigen::Matrix<Index, n_c, n_c * n_u>> negative_columns(negative_cols);
            columns = u_indices_vec.template replicate<n_c, 1>();
            negative_columns = columns;
            cols += n_c * n_c * n_u;
            negative_cols += n_c * n_c * n_u;

            /* Next, we do the derivatives with respect to time */
            Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>>(cols).fill(GetIndex::times(inds.data())(i_w));
            Eigen::Map<Eigen::Matrix<Index, n_c * n_u, 1>>(negative_cols).fill(GetIndex::times(inds.data())(i_w));
            cols += n_c * n_u;
            negative_cols += n_c * n_u;
        }
    }
};

#endif /* CONTROL_RATE_CONSTRAINTS_HEADER */