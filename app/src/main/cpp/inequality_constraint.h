#ifndef INEQUALITY_CONSTRAINT_HEADER
#define INEQUALITY_CONSTRAINT_HEADER

#include "constraint.h"

template<typename T, typename Index, typename Bool, typename Data, Index n_vars>
class InequalityConstraint : public Constraint<T, Index, Bool, Data, n_vars> {

public:

    virtual ~InequalityConstraint() {};
};

#endif /* INEQUALITY_CONSTRAINT_HEADER */
