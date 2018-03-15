#ifndef EQUALITY_CONSTRAINT_HEADER
#define EQUALITY_CONSTRAINT_HEADER

#include "constraint.h"

template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
class EqualityConstraint : public Constraint<T, Index, Bool, Data, n_vars> {

public:

    virtual ~EqualityConstraint() {};
};

#endif /* EQUALITY_CONSTRAINT_HEADER */
