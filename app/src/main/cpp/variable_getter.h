#ifndef VARIABLE_GETTER_HEADER
#define VARIABLE_GETTER_HEADER

/* iostream is just imported to get the endl operator. */
#include <iostream>
#include "Eigen/Dense"
#include "utils.h"

using std::endl;

/**
 * This class provides some easy accessors when dealing with a raw pointer
 * to variables that sits in memory like this (in column-major format):
 *
 * layout = 0:
 *                      waypoint 1,    waypoint 2, ...,    waypoint n_w
 * collocation 1:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 * collocation 2:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *     ...          [      ...    ,       ...    , ...,       ...      ]
 * collocation n_c: [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *
 * followed by the vector of times for each waypoint, [ t_1,... t_{n_w} ]
 *
 *                --------------------------------------------
 *                NOT IMPLEMENTED::::
 * layout = 1:
 *                   collocation 1, collocation 2, ..., collocation n_c
 *    waypoint 1:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *    waypoint 2:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *     ...          [      ...    ,       ...    , ...,       ...      ]
 *    waypoint n_w: [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *
 * followed by the vector of times for each waypoint, [ t_1,... t_{n_w} ]
 *
 * @tparam n_x: The size of the state
 * @tparam n_u: The size of the control input
 * @tparam n_c: The number of collocation points
 * @tparam n_w: The number of waypoints
 */
template<typename T, typename Index, Index n_x, Index n_u, Index n_c, Index n_w, Index layout = 0>
class VariableGetter {
private:

    /** Return a reference to the states and controls as a matrix.
     * This is private because using this function would make your code unportable.
     * Specifically, it would not be portable because
     * it would directly expose the underlying representation of the data.
     */
    static auto asMatrix(T *raw_ptr) {
        return Eigen::Map<Eigen::Matrix<T, (n_x + n_u) * n_c, n_w>>(raw_ptr);
    }

    Eigen::Matrix<T, n_c, n_c> derivative_coefficients;

public:

    VariableGetter(const Eigen::Matrix<T, n_c, 1> &collocation_points) {
        derivative_coefficients = lagrangeDerivativeCoefficients<Index>(collocation_points);
    }

    static const Index n_vars = ((n_x + n_u) * n_c + 1) * n_w;

    /** Compute the derivatives of all of the variables in the input raw_ptr
     * in the locations where the states and controls typically would be, saving the
     * derivatives in the specified derivative memory. Note that the pieces of memory in
     * derivative memory that hold other variables, such as time, will not be modified by
     * this function. */
    void derivatives(T *raw_ptr, T *derivative_memory) const {
        for (Index i_w = 0; i_w < n_w; ++i_w)
            varsAtWaypoint(derivative_memory, i_w) =
                    varsAtWaypoint(raw_ptr, i_w) * derivative_coefficients / times(raw_ptr)(i_w);
    }

    /** Compute the derivatives, but do not scale by time. */
    void derivativesUnscaled(T *raw_ptr, T *derivative_memory) const {
        for (Index i_w = 0; i_w < n_w; ++i_w)
            varsAtWaypoint(derivative_memory, i_w) =
                    varsAtWaypoint(raw_ptr, i_w) * derivative_coefficients;
    }

    /** Compute both the scaled and unscaled derivatives. */
    void derivatives(T *raw_ptr, T *derivative_memory, T *derivative_unscaled_memory) const {
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            varsAtWaypoint(derivative_unscaled_memory, i_w) =
                    varsAtWaypoint(raw_ptr, i_w) * derivative_coefficients;
            varsAtWaypoint(derivative_memory, i_w) =
                    varsAtWaypoint(derivative_unscaled_memory, i_w) / times(raw_ptr)(i_w);
        }
    }

    /** Return a copy of the derivative coefficients used for right multiplciation given the
     * collocation points passed to this VariableGetter during construction.
     *
     * Specifically, if fx denotes a matrix
     * containing the values of a function f evaluated at the collocation points, c:
     *
     * fx = [ f(c(0)), ... f(c(n_c-1)) ]
     *
     * and if coeffs denotes the output of this function, then we will approximately find that
     *
     * fx * coeffs = [ df/dt(c(0)), ... df/dt(c(n_c-1)) ]
     */
    Eigen::Matrix<T, n_c, n_c> rightDerivativeCoefficients() const {
        return derivative_coefficients;
    }

    /** Set all of the variables to zero */
    static void setZero(T *raw_ptr) {
        asMatrix(raw_ptr).setZero();
        times(raw_ptr).setZero();
    }

    /** Return a reference to the submatrix
     *
     * [ x[i,0], ..., x[i,n_w-1]]
     * [ u[i,0], ..., u[i,n_w-1]]
     *
     * that is, a matrix of shape (n_x+n_u) x n_w,
     * holding all of the states and controls at collocation point i_c.
     */
    static auto varsAtCollocationPoint(T *raw_ptr, Index i_c) {
        return asMatrix(raw_ptr).template middleRows<n_x + n_u>((n_x + n_u) * i_c);
    }

    /** Return a reference to the submatrix
     *
     * [ x[i,0], ..., x[i,n_w-1]]
     *
     * that is, a matrix of shape n_x x n_w,
     * holding all of the states at collocation point i_c.
     */
    static auto statesAtCollocationPoint(T *raw_ptr, Index i_c) {
        return varsAtCollocationPoint(raw_ptr, i_c).template topRows<n_x>();
    }

    /** Return a reference to
     *
     * x[i,j]
     *
     * that is, a matrix of shape n_x x 1,
     * holding the state at collocation point i_c and waypoint i_w.
     */
    static auto state(T *raw_ptr, Index i_c, Index i_w) {
        return statesAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    /** Return a reference to the submatrix
     *
     * [ u[i,0], ..., u[i,n_w-1]]
     *
     * that is, a matrix of shape n_u x n_w,
     * holding all of the controls at collocation point i_c.
     */
    static auto controlsAtCollocationPoint(T *raw_ptr, Index i_c) {
        return varsAtCollocationPoint(raw_ptr, i_c).template bottomRows<n_u>();
    }

    /** Return a reference to
     *
     * u[i,j]
     *
     * that is, a matrix of shape n_u x 1,
     * holding all of the controls at collocation point i_c and waypoint i_w.
     */
    static auto control(T *raw_ptr, Index i_c, Index i_w) {
        return controlsAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    /** Return a reference to the submatrix
     *
     * [ x[0,i], ..., x[n_c-1,i]]
     * [ u[0,i], ..., u[n_c-1,i]]
     *
     * that is, a matrix of shape (n_x + n_u) x n_c,
     * holding all of the states and controls at waypoint i_w.
     */
    static auto varsAtWaypoint(T *raw_ptr, Index i_w) {
        return Eigen::Map<Eigen::Matrix<T, n_x + n_u, n_c>>(asMatrix(raw_ptr).col(i_w).data());
    }

    /** Return a reference to the submatrix
     *
     * [ x[0,i], ..., x[n_c-1,i]]
     *
     * that is, a matrix of shape n_x x n_c,
     * holding all of the states at waypoint i_w.
     */
    static auto statesAtWaypoint(T *raw_ptr, Index i_w) {
        return varsAtWaypoint(raw_ptr, i_w).template topRows<n_x>();
    }

    /** Return a reference to the submatrix
     *
     * [ u[0,i], ..., u[n_c-1,i]]
     *
     * that is, a matrix of shape n_u x n_c,
     * holding all of the controls at waypoint i_w.
     */
    static auto controlsAtWaypoint(T *raw_ptr, Index i_w) {
        return varsAtWaypoint(raw_ptr, i_w).template bottomRows<n_u>();
    }

    /** Return a reference to the submatrix
     *
     * [ t[0], ..., t[n_w-1]]
     *
     * that is, a matrix of shape 1 x n_w,
     * holding all of the times.
     */
    static auto times(T *raw_ptr) {
        return Eigen::Map<Eigen::Matrix<T, 1, n_w>>(raw_ptr + (n_x + n_u) * n_c * n_w);
    }

    /** Return a nice formatted string of all of the variables */
    static std::string asString(T *raw_vars) {
        std::stringstream out;

        /* First, write the times to the string  */
        out << endl;
        out << endl;
        out << "Times: " << times(raw_vars) << endl;

        out << "----------------------------" << endl;
        out << endl;
        out << "Controls: " << endl;
        out << endl;
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            out << "Collocation point " << i_c << endl;
            out << controlsAtCollocationPoint(raw_vars, i_c) << endl;
        }
        out << endl;
        out << "----------------------------" << endl;

        out << endl;
        out << "States: " << endl;
        out << endl;
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            out << "Collocation point " << i_c << endl;
            out << statesAtCollocationPoint(raw_vars, i_c) << endl;
        }
        out << endl;
        out << "----------------------------" << endl;
        return out.str();
    }
};

#endif /* VARIABLE_GETTER_HEADER */