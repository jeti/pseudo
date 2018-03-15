#ifndef INITIAL_STATE_CONSTRAINTS_HEADER
#define INITIAL_STATE_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"
#include "problem_data.h"

/**
 * This set of constraints simply ensures that the
 * estimate of the state at collocation point 0, waypoint 0 is
 * equal to the value of the initial state in the ProblemData.
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class InitialStateConstraints: public EqualityConstraint<T,
                                                         Index,
                                                         Bool,
                                                         Data,
                                                         VariableGetter<T,
                                                                        Index,
                                                                        n_x,
                                                                        n_u,
                                                                        n_c,
                                                                        n_w>::n_vars> {
private:

    static const Index n_constraints = n_x;
    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
    using GetIndex = VariableGetter<Index, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<T, n_x, 1>>;
    using MapIndex = Eigen::Map<Eigen::Matrix<Index, n_x, 1>>;
    using PD = ProblemData<T, Index, Bool, Data, n_x, n_u, n_c, n_w>;

public:

    ~InitialStateConstraints() {};

    Index getNumberOfConstraints() const override {
        return n_constraints;
    }

    Index evaluate(T *x,
                   Bool new_x,
                   T *g,
                   Data *data,
                   Index derivatives_in_data = 0) override {
        PD *pd = static_cast<PD*>(data);
        Map G(g);
        G = Get::state(x, 0, 0) - pd->initial_state;
        return derivatives_in_data;
    }

    /* The constraint just does n_x binary comparisons between
     * variables x and the initial state. So we have n_x terms in the
     * jacobian, all of which are 1.
     */
    Index evaluateJacobian(T *x,
                           Bool new_x,
                           T *dgdx,
                           Data *data,
                           Index derivatives_in_data = 0) override {
        for (Index i = 0; i < n_constraints; ++i) {
            dgdx[i] = 1;
        }
        return derivatives_in_data;
    }

    /* The jacobian just does n_x
     * binary comparisons between variables. */
    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) override {
        /* The rows are easy. Only a single variable will enter into
         * each constraint. So we just need to count from 0, .., # of constraints */
        this->nonzero_rows.resize(n_constraints);
        for (Index i = 0; i < n_constraints; ++i) {
            this->nonzero_rows[i] = i;
        }

        /* To get the nonzero columns, we essentially replace
         * x with a list of indices in the constraint function, and then
         * see which values are pulled off. */
        Eigen::Matrix<Index, Get::n_vars, 1> inds;
        for (Index i = 0; i < Get::n_vars; ++i)
            inds[i] = i;

        this->nonzero_cols.resize(n_constraints);
        MapIndex nonzeros(this->nonzero_cols.data());
        nonzeros = GetIndex::state(inds.data(), 0, 0);
    }
};

#endif /* COLLOCATION_CONSTRAINTS_HEADER */