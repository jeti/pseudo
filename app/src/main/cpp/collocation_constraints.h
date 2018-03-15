#ifndef COLLOCATION_CONSTRAINTS_HEADER
#define COLLOCATION_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * We are redundantly estimating the state and control at the
 * ends of the waypoints since the collocation points go from
 * 0...1 (including the endpoints).
 * This gives us (n_x + n_u) * (n_w-1) conditions that must be checked.
 * Note that we have (n_w-1) here because we don't have redundant
 * estimates at the initial time.
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class CollocationConstraints
        : public EqualityConstraint<T, Index, Bool, Data, VariableGetter<T, Index, n_x, n_u, n_c, n_w>::n_vars> {
private:

    static const Index n_constraints = (n_x + n_u) * (n_w - 1);
    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
    using GetIndex = VariableGetter<Index, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<T, n_x + n_u, n_w - 1 >>;
    using MapIndex = Eigen::Map<Eigen::Matrix<Index, n_x + n_u, n_w - 1 >>;

public:

    ~CollocationConstraints() {};

    Index getNumberOfConstraints() const override {
        return n_constraints;
    }

    /** Evaluate the constraint at x and store the values at in g. */
    Index evaluate(T *x,
                   Bool new_x,
                   T *g,
                   Data *data,
                   Index derivatives_in_data = 0) override {

        /* Get all of the values at the first and last collocation points */
        auto c_0 = Get::varsAtCollocationPoint(x, 0);
        auto c_end = Get::varsAtCollocationPoint(x, n_c - 1);

        /* We are going to compute c_0 (of waypoint i) - c_end (of waypoint i-1)
         * This will be a matrix of size (n_x + n_u) x (n_w-1) */
        Map G(g);
        G = c_0.template rightCols<n_w - 1>() - c_end.template leftCols<n_w - 1>();
        return derivatives_in_data;
    }

    /* The constraint just does (n_x + n_u) * (n_w-1) binary comparisons between
     * variables x. So (n_x + n_u) * (n_w-1) terms in the jacobian will be 1,
     * and another (n_x + n_u) * (n_w-1) terms will be minus 1.
     *
     * For instance, this constraint will check that
     * the state at collocation point 1, waypoint i is equal to
     * the state at collocation point n_c, waypoint i-1
     *
     * x[0,i] - x[n_c-1,i] = 0
     */
    Index evaluateJacobian(T *x,
                           Bool new_x,
                           T *dgdx,
                           Data *data,
                           Index derivatives_in_data = 0) override {
        for (Index i = 0; i < n_constraints; ++i) {
            dgdx[i] = 1;
            dgdx[i + n_constraints] = -1;
        }
        return derivatives_in_data;
    }

    /* The jacobian just does (n_x + n_u) * (n_w-1)
     * binary comparisons between variables. */
    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) override {
        /* First, we get the indices of all of the "positive entries", then
         * we get the indices of the variables we are subtracting off.
         * So the nonzero rows will be
         * 0, ..., n_constraints-1, 0, ..., n_constraints-1 */
        this->nonzero_rows.resize(2 * n_constraints);
        for (Index i = 0; i < n_constraints; ++i) {
            this->nonzero_rows[i] = i;
            this->nonzero_rows[i + n_constraints] = i;
        }

        /* To get the nonzero columns, we replace essentially replace
         * x with a list of indices in the constraint function, and then
         * see which values are pulled off. */
        Eigen::Matrix<Index, Get::n_vars, 1> inds;
        for (Index i = 0; i < Get::n_vars; ++i)
            inds(i) = i;

        /* First, we set the positive indices */
        this->nonzero_cols.resize(2 * n_constraints);
        MapIndex positives(this->nonzero_cols.data());
        MapIndex negatives(this->nonzero_cols.data() + n_constraints);

        positives = GetIndex::varsAtCollocationPoint(inds.data(), 0).template rightCols<n_w - 1>();
        negatives = GetIndex::varsAtCollocationPoint(inds.data(), n_c - 1).template leftCols<n_w - 1>();
    }
};

#endif /* COLLOCATION_CONSTRAINTS_HEADER */