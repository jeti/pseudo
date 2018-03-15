#ifndef FUSED_CONSTRAINT_HEADER
#define FUSED_CONSTRAINT_HEADER

#include <vector>
#include "constraint.h"
#include "equality_constraint.h"
#include "inequality_constraint.h"

template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
class FusedConstraint : public Constraint<T, Index, Bool, Data, n_vars> {

private:

    using Indices = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using Values = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Index n_equalities;
    Index n_inequalities;
    Index n_constraints;
    std::vector<EqualityConstraint<T, Index, Bool, Data, n_vars> *> equality_constraints;
    std::vector<InequalityConstraint<T, Index, Bool, Data, n_vars> *> inequality_constraints;
    const T equality_lower_bound;
    const T equality_upper_bound;
    const T inequality_lower_bound;
    const T inequality_upper_bound;

    /** Concatenate the nonzero rows and cols from each of the constraints.
     * The caveat is that the cols are unchanged, but we will be stacking up
     * the constraints vertically, so the rows need to be offset to reflect their
     * actual position in the constraint jacobian */
    void appendConstraint(Constraint<T, Index, Bool, Data, n_vars> *constraint,
                          Index index_offset,
                          Index row_offset) {

        /* Concatenate the nonzero cols directly to the list */
        Indices nz_cols = constraint->getNonzeroJacobianCols();
        for (Index i = 0; i < nz_cols.size(); ++i)
            this->nonzero_cols(i + index_offset) = nz_cols(i);

        /* Concatenate the nonzero rows to the list, making sure to offset them. */
        Indices nz_rows = constraint->getNonzeroJacobianRows();
        for (Index i = 0; i < nz_rows.size(); ++i)
            this->nonzero_rows(i + index_offset) = nz_rows(i) + row_offset;
    }

public:

    FusedConstraint(const std::vector<EqualityConstraint<T,
            Index,
            Bool,
            Data,
            n_vars> *> &equality_constraints,
                    const std::vector<InequalityConstraint<T,
                            Index,
                            Bool,
                            Data,
                            n_vars> *> &inequality_constraints,
                    const T equality_lower_bound = -1e-10,
                    const T equality_upper_bound = +1e-10,
                    const T inequality_lower_bound = -1e10,
                    const T inequality_upper_bound = +1e-10)
            : equality_constraints(equality_constraints),
              inequality_constraints(inequality_constraints),
              equality_lower_bound(equality_lower_bound),
              equality_upper_bound(equality_upper_bound),
              inequality_lower_bound(inequality_lower_bound),
              inequality_upper_bound(inequality_upper_bound) {

        /* Calculate the number of constraints. */
        n_equalities = 0;
        for (auto &constraint: this->equality_constraints) {
            n_equalities += constraint->getNumberOfConstraints();
        }
        n_inequalities = 0;
        for (auto &constraint: this->inequality_constraints) {
            n_inequalities += constraint->getNumberOfConstraints();
        }
        n_constraints = n_equalities + n_inequalities;
    }

    ~FusedConstraint() {};

    Index getNumberOfEqualities() const {
        return n_equalities;
    }

    Index getNumberOfInequalities() const {
        return n_inequalities;
    }

    Index getNumberOfConstraints() const {
        return n_constraints;
    }

    Index evaluate(T *x,
                   Bool new_x,
                   T *g,
                   Data *data,
                   Index derivatives_in_data = 0) {
        for (auto &constraint: equality_constraints) {
            derivatives_in_data = constraint->evaluate(x, new_x, g, data, derivatives_in_data);
            g += constraint->getNumberOfConstraints();
        }
        for (auto &constraint: inequality_constraints) {
            derivatives_in_data = constraint->evaluate(x, new_x, g, data, derivatives_in_data);
            g += constraint->getNumberOfConstraints();
        }
        return derivatives_in_data;
    }

    Index evaluateJacobian(T *x,
                           Bool new_x,
                           T *dgdx,
                           Data *data,
                           Index derivatives_in_data = 0) {
        for (auto &constraint: equality_constraints) {
            derivatives_in_data = constraint->evaluateJacobian(x, new_x, dgdx, data, derivatives_in_data);
            dgdx += constraint->getNumberOfJacobianNonzeros();
        }
        for (auto &constraint: inequality_constraints) {
            derivatives_in_data = constraint->evaluateJacobian(x, new_x, dgdx, data, derivatives_in_data);
            dgdx += constraint->getNumberOfJacobianNonzeros();
        }
        return derivatives_in_data;
    }

    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) {

        /* The first thing we ned to do is to generate the sparsity structure for
         * each of the constraints. We do this first because we want to know
         * how many nonzero rows and columns we will need to store. */
        Index jacobian_nonzeros = 0;
        for (auto &constraint: equality_constraints) {
            constraint->generateSparsityStructure(x, new_x, data);
            jacobian_nonzeros += constraint->getNumberOfJacobianNonzeros();
        }
        for (auto &constraint: inequality_constraints) {
            constraint->generateSparsityStructure(x, new_x, data);
            jacobian_nonzeros += constraint->getNumberOfJacobianNonzeros();
        }

        /* Now we know how many nonzero values we will have.
         * Let's allocate space, and then start filling the rows and cols.*/
        this->nonzero_rows.resize(jacobian_nonzeros);
        this->nonzero_cols.resize(jacobian_nonzeros);
        Index index_offset = 0;
        Index row_offset = 0;
        for (auto &constraint: equality_constraints) {
            appendConstraint(constraint, index_offset, row_offset);
            index_offset += constraint->getNumberOfJacobianNonzeros();
            row_offset += constraint->getNumberOfConstraints();
        }
        for (auto &constraint: inequality_constraints) {
            appendConstraint(constraint, index_offset, row_offset);
            index_offset += constraint->getNumberOfJacobianNonzeros();
            row_offset += constraint->getNumberOfConstraints();
        }
    }

    Values getLowerBound() {
        Values bound(getNumberOfConstraints());
        bound.topRows(getNumberOfEqualities()).fill(equality_lower_bound);
        bound.bottomRows(getNumberOfInequalities()).fill(inequality_lower_bound);
        return bound;
    }

    Values getUpperBound() {
        Values bound(getNumberOfConstraints());
        bound.topRows(getNumberOfEqualities()).fill(equality_upper_bound);
        bound.bottomRows(getNumberOfInequalities()).fill(inequality_upper_bound);
        return bound;
    }
};

#endif /* FUSED_CONSTRAINT_HEADER */