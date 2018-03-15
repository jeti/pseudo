/* Properly define cout and endl for your system */
#include "cout.h"

/* IPOPT */
#include "IpStdCInterface.h"

/* STL libraries that we are using */
#include <vector>
#include <assert.h>
#include <chrono>

/* Our classes */
#include "constants.h"
#include "custom_types.h"
#include "utils.h"
#include "equality_constraint.h"
#include "inequality_constraint.h"
#include "fused_constraint.h"
#include "collocation_constraints.h"
#include "waypoint_constraint.h"
#include "waypoint_constraints.h"
#include "control_rate_constraints.h"
#include "initial_state_constraints.h"
#include "smooth_control_constraints.h"
#include "dynamics_constraints.h"
#include "jacobian_utils.h"
#include "variable_getter.h"

/* Eigen */
#include "Eigen/Dense"

/* All of the aliases we use a lot */
using std::endl;

using Eigen::Dynamic;
using Eigen::IOFormat;
using Eigen::Map;
using Eigen::Matrix;

using Indices = Matrix<Index, Dynamic, 1>;
using Values = Matrix<T, Dynamic, 1>;
using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
using PD = ProblemData<T, Index, Bool, Data, n_x, n_u, n_c, n_w>;

template <typename T>
void log_state(T *vars, bool waypoints = true,
               const IOFormat &format = IOFormat(4, 0, " ", "\n", "", "", "", ""))
{

    /* First, write the times to the string  */
    cout << endl
         << endl;
    cout << "Times: ";
    auto t = Get::times(vars);
    cout << t.format(format);
    cout << endl;

    cout << "----------------------------";
    cout << endl
         << endl;
    cout << "Controls: ";
    cout << endl
         << endl;
    if (waypoints)
    {
        for (Index i_w = 0; i_w < n_w; ++i_w)
        {
            cout << "Waypoint " << i_w << endl;
            auto u = Get::controlsAtWaypoint(vars, i_w);
            cout << u.format(format);
            cout << endl
                 << endl;
        }
    }
    else
    {
        for (Index i_c = 0; i_c < n_c; ++i_c)
        {
            cout << "Collocation point " << i_c << endl;
            auto u = Get::controlsAtCollocationPoint(vars, i_c);
            cout << u.format(format);
            cout << endl
                 << endl;
        }
    }
    cout << endl;
    cout << "----------------------------";
    cout << endl
         << endl;
    cout << "States: ";
    cout << endl
         << endl;
    if (waypoints)
    {
        for (Index i_w = 0; i_w < n_w; ++i_w)
        {
            cout << "Waypoint " << i_w << endl;
            auto x = Get::statesAtWaypoint(vars, i_w);
            cout << x.format(format);
            cout << endl
                 << endl;
        }
    }
    else
    {
        for (Index i_c = 0; i_c < n_c; ++i_c)
        {
            cout << "Collocation point " << i_c << endl;
            auto x = Get::statesAtCollocationPoint(vars, i_c);
            cout << x.format(format);
            cout << endl
                 << endl;
        }
    }
    cout << endl;
    cout << "----------------------------";
    cout << endl;
}

class Strings
{
  public:
    std::vector<std::vector<char>> strings;

    char *operator()(const std::string &str)
    {
        std::vector<char> chars(str.begin(), str.end());
        chars.push_back('\0');
        strings.push_back(chars);
        return strings.back().data();
    }
};

/**
 * The objective cost is to minimize the final time.
 * Which is obtained by summing up the times required to go between waypoints.
 */
extern "C" Bool eval_f(Index n_vars_,
                       T *x,
                       Bool new_x,
                       T *obj_value,
                       Data *data)
{
    assert(n_vars == n_vars_);
    *obj_value = Get::times(x).sum();
    return TRUE;
}

/** The gradient of the objective function calculated using numerical differentiation.
 * Note: If you change the objective cost, then you can no longer use the simple jacobian!!!
 */
extern "C" Bool eval_grad_f(Index n_vars_,
                            T *x,
                            Bool new_x,
                            T *grad_f,
                            Data *data)
{
    assert(n_vars == n_vars_);
    Get::setZero(grad_f);
    Get::times(grad_f).setOnes();
    return TRUE;
}

extern "C" Bool eval_g(Index n_vars_,
                       T *x,
                       Bool new_x,
                       Index n_constraints_,
                       T *g,
                       Data *data)
{

    PD *pd = static_cast<PD *>(data);
    assert(n_vars == n_vars_);
    assert(pd->fused_constraint.getNumberOfConstraints() == n_constraints_);
    pd->fused_constraint.evaluate(x, new_x, g, data);
    return TRUE;
}

extern "C" Bool eval_jac_g(Index n_vars_,
                           T *x,
                           Bool new_x,
                           Index n_constraints_,
                           Index jacobian_nonzeros_,
                           Index *nonzero_rows,
                           Index *nonzero_cols,
                           T *dgdx,
                           Data *data)
{
    PD *pd = static_cast<PD *>(data);
    assert(n_vars == n_vars_);
    assert(pd->fused_constraint.getNumberOfConstraints() == n_constraints_);
    assert(pd->fused_constraint.getNumberOfJacobianNonzeros() == jacobian_nonzeros_);
    if (dgdx)
    {
        /* If dgdx is not a null pointer, evaluate. */
        pd->fused_constraint.evaluateJacobian(x, new_x, dgdx, data);
    }
    else
    {
        /* Otherwise, IPOPT wants our sparsity structure. */
        Map<Matrix<Index, Dynamic, 1>> rows(nonzero_rows, jacobian_nonzeros_);
        rows = pd->fused_constraint.getNonzeroJacobianRows();

        Map<Matrix<Index, Dynamic, 1>> cols(nonzero_cols, jacobian_nonzeros_);
        cols = pd->fused_constraint.getNonzeroJacobianCols();
    }
    return TRUE;
}

/**
 * We are L-BFGS method for estimating the inverse of Hessian. So the Hessian is not computed.
 */
extern "C" Bool eval_h(Index n_vars_, T *x, Bool new_x, T obj_factor,
                       Index n_constraints_, T *lambda, Bool new_lambda,
                       Index hessian_nonzeros, Index *iRow, Index *jCol,
                       T *values, Data *data)
{
    PD *pd = static_cast<PD *>(data);
    assert(n_vars == n_vars_);
    assert(pd->fused_constraint.getNumberOfConstraints() == n_constraints_);
    return FALSE;
}

extern "C" Bool intermediate_cb(Index alg_mod, Index iter_count, T obj_value,
                                T inf_pr, T inf_du, T mu, T d_norm,
                                T regularization_size, T alpha_du,
                                T alpha_pr, Index ls_trials, Data *data)
{
    printf("Testing intermediate callback in iteration %d\n", iter_count);
    if (inf_pr < 1e-4)
        return FALSE;
    return TRUE;
}

/* The "main" function that will be called from Java */
#ifdef __ANDROID__

extern "C" JNIEXPORT jdouble JNICALL
Java_io_pcess_trajectory_1optimization_MainActivity_optimizetrajectory(JNIEnv *env,
                                                                       jobject object)
{

#else

int main()
{

#endif

    /*
     * ----------------------------------------------
     *
     * Waypoints
     * TODO: Pass from Java
     *
     * ----------------------------------------------
     */
    Matrix<T, n_x, n_w> waypoints;
    waypoints.setZero();
    waypoints.col(0) << 2.0, 2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 2)
        waypoints.col(1) << 4.0, 2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 3)
        waypoints.col(2) << 8.0, 0.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 4)
        waypoints.col(3) << 4.0, -2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 5)
        waypoints.col(4) << 2.0, -2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 6)
        waypoints.col(5) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /*
     * ----------------------------------------------
     *
     * Initial State
     * TODO: Pass from Java
     *
     * ----------------------------------------------
     */
    Matrix<T, n_x, 1> initial_state;
    initial_state.setZero();

    /*
     * ----------------------------------------------
     *
     * Collocation points
     *
     * ----------------------------------------------
     */
    Matrix<T, n_c, 1> collocation_points = generateCollocationPoints<T, Index, n_c>();

    /*
     * ----------------------------------------------
     *
     * Upper and lower bounds for the variables
     *
     * ----------------------------------------------
     */
    Matrix<T, n_vars, 1> lower_bound;
    Matrix<T, n_vars, 1> upper_bound;

    /* State bounds. We will set the same state bounds for each collocation point and waypoint.  */
    Matrix<T, n_x, 1> x_min;
    x_min << -2e19, -2e19, -2e19, -2e19, -2e19, -2e19;

    Matrix<T, n_x, 1> x_max;
    x_max << 2e19, 2e19, 0, 2e19, 2e19, 2e19;

    for (Index i_w = 0; i_w < n_w; ++i_w)
    {
        auto x_lower = Get::statesAtWaypoint(lower_bound.data(), i_w);
        x_lower = x_min.replicate<1, n_c>();
        auto x_upper = Get::statesAtWaypoint(upper_bound.data(), i_w);
        x_upper = x_max.replicate<1, n_c>();
    }

    /* Control bounds */
    Matrix<T, n_u, 1> u_min;
    u_min << 0, -30 * M_PI / 180, -30 * M_PI / 180, -2 * 360 * M_PI / 180;

    Matrix<T, n_u, 1> u_max;
    u_max << 2 * 9.91, 30 * M_PI / 180, 30 * M_PI / 180, 2 * 360 * M_PI / 180;

    for (Index i_w = 0; i_w < n_w; ++i_w)
    {
        auto u_lower = Get::controlsAtWaypoint(lower_bound.data(), i_w);
        u_lower = u_min.template replicate<1, n_c>();
        auto u_upper = Get::controlsAtWaypoint(upper_bound.data(), i_w);
        u_upper = u_max.template replicate<1, n_c>();
    }

    /* Time bounds */
    const T time_min = 0;
    const T time_max = 10;
    Get::times(lower_bound.data()).fill(time_min);
    Get::times(upper_bound.data()).fill(time_max);

    /*
     * ----------------------------------------------
     *
     * Initial guess
     *
     * ----------------------------------------------
     */
    Matrix<T, n_vars, 1> initial_guess;
    initial_guess.setZero();

    /* Set the times to the specified value. */
    T time = 1;
    Get::times(initial_guess.data()).fill(time);

    /* Now compute the difference between each of the waypoints.
     * The first column is the first waypoint - the initial state.
     * The rest are waypoint i - waypoint (i-1)*/
    Matrix<T, n_x, n_w> differences;
    differences << waypoints.col(0) - initial_state,
        waypoints.template rightCols<n_w - 1>() - waypoints.template leftCols<n_w - 1>();

    /* Now we can interpolate the values for the state using the formula
     *
     * interpolated = final - (1 - collocation_point) * ( final - initial )
     */
    Matrix<T, n_c, 1> dc = -collocation_points.array() + 1;
    for (Index i_c = 0; i_c < n_c; ++i_c)
    {
        auto x = Get::statesAtCollocationPoint(initial_guess.data(), i_c);
        x = waypoints - dc(i_c) * differences;
    }

    /* Note that we are leaving the initial control guesses as zeros. */

    /*
     * ----------------------------------------------
     *
     * Constraints
     *
     * ----------------------------------------------
     */
    CollocationConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w> collocation_constraints;
    InitialStateConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w> initial_state_constraints;
    SmoothControlConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w>
        smooth_control_constraints;
    DynamicsConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w> dynamics_constraints;

    Matrix<T, n_u, 1> u_dot_max;
    T max_angular_rate = 30 * M_PI / 180;
    u_dot_max << 20, max_angular_rate, max_angular_rate, max_angular_rate;
    Matrix<T, n_u, 1> u_dot_min = -u_dot_max;
    ControlRateConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w> control_rate_constraints(u_dot_min, u_dot_max);

    /* Create constraints for each of the positions */
    WaypointConstraint<T, Index, Bool, Data, n_x, n_u, n_c, n_w> waypoint_x(0, waypoints.row(0));
    WaypointConstraint<T, Index, Bool, Data, n_x, n_u, n_c, n_w> waypoint_y(1, waypoints.row(1));
    WaypointConstraint<T, Index, Bool, Data, n_x, n_u, n_c, n_w> waypoint_z(2, waypoints.row(2));
    WaypointConstraints<T, Index, Bool, Data, n_x, n_u, n_c, n_w> waypoint_constraints;

    /* Equality constraints */
    std::vector<EqualityConstraint<T, Index, Bool, Data, n_vars> *> equality_constraints;
    equality_constraints.push_back(&initial_state_constraints);
    equality_constraints.push_back(&smooth_control_constraints);
    bool use_all_waypoints = false;
    if (use_all_waypoints)
    {
    equality_constraints.push_back(&waypoint_constraints);
    }
    else
    {
        equality_constraints.push_back(&waypoint_x);
        equality_constraints.push_back(&waypoint_y);
        equality_constraints.push_back(&waypoint_z);
    }
    equality_constraints.push_back(&collocation_constraints);
    equality_constraints.push_back(&dynamics_constraints);

    /* Inequality constraints */
    std::vector<InequalityConstraint<T, Index, Bool, Data, n_vars> *> inequality_constraints;
    inequality_constraints.push_back(&control_rate_constraints);

    /* Now fuse together all of our constraints */
    FusedConstraint<T, Index, Bool, Data, n_vars> fc(equality_constraints,
                                                     inequality_constraints);

    /* Generate the constraint lower and upper bounds */
    Matrix<T, Dynamic, 1>
        constraint_lower_bound = fc.getLowerBound();
    Matrix<T, Dynamic, 1>
        constraint_upper_bound = fc.getUpperBound();

    /*
     * ----------------------------------------------
     *
     * Problem Data
     *
     * ----------------------------------------------
     */
    /* The problem data will need a VariableGetter instance,
     * which requires collocation points so that it can generate derivatives. */
    Get getter(collocation_points);

    /* Construct the problem data */
    PD problem_data(waypoints, initial_state, getter, fc);

    /* This may seem a bit roundabout, but I want to access everything just like it would be in the constraint functions */
    Data *data = &problem_data;
    PD *pd = static_cast<PD *>(data);

    /*
     * ----------------------------------------------
     *
     * IPOPT Problem definition
     *
     * ----------------------------------------------
     */
    /* Generate the sparsity structure. This is necessary so that we know how many nonzeros
     * the jacobian has (which is a parameter required by the IPOPT problem constructor) */
    pd->fused_constraint.generateSparsityStructure(initial_guess.data(), true, data);

    /* T of nonzeros in the Hessian of the Lagrangian (lower or upper triangualar part only).
     * Since we are using BFGS, we aren't computing the hessian, so we don't need this. */
    const Index hessian_nonzeros = 0;

    /* Indexing style for matrices... 0 denotes C-style counting from 0. */
    const Index index_style = 0;

    /* Create the IpoptProblem */
    IpoptProblem nlp = CreateIpoptProblem(n_vars,
                                          lower_bound.data(),
                                          upper_bound.data(),
                                          pd->fused_constraint.getNumberOfConstraints(),
                                          constraint_lower_bound.data(),
                                          constraint_upper_bound.data(),
                                          pd->fused_constraint.getNumberOfJacobianNonzeros(),
                                          hessian_nonzeros,
                                          index_style,
                                          &eval_f,
                                          &eval_g,
                                          &eval_grad_f,
                                          &eval_jac_g,
                                          &eval_h);

    const bool check_jacobians = false;
    if (check_jacobians)
    {

        Matrix<T, n_vars, 1> xx;
        for (int i = 0; i < n_vars; ++i)
            xx(i) = i + 1;
        auto times = Get::times(xx.data());
        for (int i = 0; i < n_w; ++i)
        {
            times(i) = 1;
        }
        Bool new_x = true;
        using Mat = Matrix<T, Dynamic, n_vars>;
        Mat full = full_jacobian(pd->fused_constraint, xx.data(), new_x, data);
        Mat fd = finite_difference(pd->fused_constraint, xx.data(), new_x, data);
        Mat check = check_jacobian(pd->fused_constraint, xx.data(), new_x, data);
        if (verbose)
        {
            bool ints = true;
            if (ints)
            {
                cout << "derivative coefficients" << endl
                     << endl
                     << pd->getter.rightDerivativeCoefficients() << endl
                     << "-------------------------------------" << endl;
                cout << "full_jacobian: " << endl
                     << endl
                     << full.cast<Index>() << endl
                     << "-------------------------------------" << endl;
                cout << "finite_difference: " << endl
                     << endl
                     << fd.cast<Index>() << endl
                     << "-------------------------------------" << endl;
                cout << "check_jacobian: " << endl
                     << endl
                     << check.cast<Index>() << endl
                     << "-------------------------------------" << endl;
            }
            else
            {
                IOFormat format(2, 0, " ", "\n", "", "", "", "");
                cout << "derivative coefficients" << endl
                     << endl
                     << pd->getter.rightDerivativeCoefficients() << endl
                     << "-------------------------------------" << endl;
                cout << "full_jacobian: " << endl
                     << endl
                     << full.format(format) << endl
                     << "-------------------------------------" << endl;
                cout << "finite_difference: " << endl
                     << endl
                     << fd.format(format) << endl
                     << "-------------------------------------" << endl;
                cout << "check_jacobian: " << endl
                     << endl
                     << check.format(format) << endl
                     << "-------------------------------------" << endl;
            }
        cout << "nonzero_rows" << endl
             << endl
             << pd->fused_constraint.getNonzeroJacobianRows() << endl
             << endl;
        cout << "nonzero_cols" << endl
             << endl
             << pd->fused_constraint.getNonzeroJacobianCols() << endl
             << endl;
        }
        cout << "max jacobian error: " << endl
             << endl
             << check.cwiseAbs().maxCoeff() << endl
             << "-------------------------------------"
             << endl;
        return 0;
    }

    /*
     * ----------------------------------------------
     *
     * IPOPT Solve
     *
     * ----------------------------------------------
     */
    /* Set some options for the solver */
    Strings str;
    AddIpoptNumOption(nlp, str("tol"), 1e-3);
    AddIpoptIntOption(nlp, str("max_iter"), 500);
    AddIpoptStrOption(nlp, str("mu_strategy"), str("adaptive"));
    AddIpoptStrOption(nlp, str("hessian_approximation"), str("limited-memory"));
    AddIpoptIntOption(nlp, str("print_level"), 5);
    // AddIpoptStrOption(nlp, str("warm_start_init_point"), str("yes"));

    /* objective value at the solution */
    T obj = -1;

    /* Allocate space to store the bound multipliers at the solution.
     * If you are overflowing the stack, then you can malloc the memory,
     * but then you must free it after the FreeIpoptProblem call (and before returning). */
    /* constraint multipliers at the solution */
    Matrix<T, Dynamic, 1> mult_g(pd->fused_constraint.getNumberOfConstraints());

    /* lower bound multipliers at the solution */
    Matrix<T, n_vars, 1> mult_x_L;

    /* upper bound multipliers at the solution */
    Matrix<T, n_vars, 1> mult_x_U;

    /* Set the callback method for intermediate user-control.  This is
     * not required, just gives you some intermediate control in case
     * you need it. */
    /* SetIntermediateCallback(nlp, intermediate_cb); */

    /* Solve the problem */
    Matrix<T, n_vars, 1> solution = initial_guess;

    enum ApplicationReturnStatus status;
    long long elapsed = 0;
    const int iterations = 10;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        solution = initial_guess;
        auto start = std::chrono::high_resolution_clock::now();
        status = IpoptSolve(nlp,
                            solution.data(),
                            NULL,
                            &obj,
                            mult_g.data(),
                            mult_x_L.data(),
                            mult_x_U.data(),
                            data);
        auto finish = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    }
    cout << "Elapsed seconds for " << iterations << " calls: " << (elapsed / 1e9) << endl;

    /*
     * ----------------------------------------------
     *
     * Print the solution and other diagnostics
     *
     * ----------------------------------------------
     */
    /* Log the solution.
     * TODO: Ideally, these would be returned to the Java side. */
    cout << endl
         << "Status = " << status << endl
         << "Cost   = " << obj << endl
         << endl;

    if (verbose)
    {
    log_state(solution.data());

    /* Show us the min and max value of all of the constraints at the solution */
    for (auto &constraint : equality_constraints)
    {
        Bool new_x = true;
        Matrix<T, Dynamic, 1> g(constraint->getNumberOfConstraints());
        constraint->evaluate(solution.data(), new_x, g.data(), data);
        cout << "constraint min = " << g.minCoeff() << ", max = " << g.maxCoeff() << endl;
    }
    }

    /* Free allocated memory */
    FreeIpoptProblem(nlp);

    /* Beep when finished */
    cout << '\a';
    return obj;
}