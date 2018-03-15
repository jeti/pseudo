#ifndef DYNAMICS_CONSTRAINTS_HEADER
#define DYNAMICS_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"
#include "problem_data.h"

/**
 * This set of constraints simply ensures that the dynamics are actually satisfied,
 * that is, that the estimated derivatives (using Lagrange interpolation polynomials)
 * equal the actual dynamics. These should hold at every collocation and waypoint for
 * each state variables, yielding n_c * n_x * n_w constraints.
 *
 * NOTE: This function assumes that the dynamics will be placing terms in the Jacobian that
 * are in different locations than the Lagrange interpolation polynomials.
 * We checked that this condition holds for our case, but you will need to modify the code
 * if that is not true.
 */
template<typename T, typename Index, typename Bool, typename Data, Index n_x, Index n_u, Index n_c, Index n_w>
class DynamicsConstraints
        : public EqualityConstraint<T, Index, Bool, Data, VariableGetter<T, Index, n_x, n_u, n_c, n_w>::n_vars> {
private:

    using Get = VariableGetter<T, Index, n_x, n_u, n_c, n_w>;
    using GetIndex = VariableGetter<Index, Index, n_x, n_u, n_c, n_w>;
    using JacEntry = Eigen::Map<Eigen::Array<T, n_c, 1>>;
    using JacLagrangeEntry = Eigen::Map<Eigen::Array<T, n_c, 1>>;
    using Map = Eigen::Map<Eigen::Matrix<T, n_x, n_c >>;
    using PD = ProblemData<T, Index, Bool, Data, n_x, n_u, n_c, n_w>;
    using Row = Eigen::Array<T, 1, n_c>;

    static const Index n_constraints = n_c * n_x * n_w;
    const T mass = 1.0;
    const T gravity = 9.81;
    const T mass_gravity = mass * gravity;

    Eigen::Matrix<T, n_c * n_c, 1> negative_derivative_coefficients;

public:

    DynamicsConstraints() {
        static_assert(n_x == 6, "This function is only valid for states of size 6");
        static_assert(n_u == 4, "This function is only valid for controls of size 4");
    }

    ~DynamicsConstraints() {};

    Index getNumberOfConstraints() const override {
        return n_constraints;
    }

    /**
     * The dynamics have the state vector
     *
     * x = [ px, py, pz, vx, vy, vz  ]
     */
    void dynamics(T *x, T *dx, Index waypoint_index) {

        Eigen::Ref<Eigen::Matrix<T, n_u, n_c>> u = Get::controlsAtWaypoint(x, waypoint_index);
        auto thrust = u.row(0).array();
        auto phi = u.row(1).array();
        auto theta = u.row(2).array();
        auto psi = u.row(3).array();

        Eigen::Map<Eigen::Matrix<T, n_x, n_c >> dX(dx);
        dX.topRows(3) = Get::statesAtWaypoint(x, waypoint_index).bottomRows(3).eval();
        dX.row(3) = -thrust * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta));
        dX.row(4) = thrust * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta));
        dX.row(5) = -thrust * cos(phi) * cos(theta) + mass_gravity;
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

        /* For each waypoint, compare the dynamics with the Lagrange interpolation polynomial */
        T *dx = pd->getDerivatives(1);
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            dynamics(x, g, i_w);
            Map G(g);
            G -= Get::statesAtWaypoint(dx, i_w);
            g += n_x * n_c;
        }
        return derivatives_in_data;
    }

    /** The jacobian of the dynamics at a specific waypoint and collocation point
     * is not dependent on the states or controls at other waypoints and/or
     * collocation points. So this function returns the jacobian
     *
     * [ dx_dot_1 / dx_1, ... , dx_dot_1 / dx_6, dx_dot_1 / du_1, ..., , dx_dot_1 / du_4 ]
     * [ ............................................................................... ]
     * [ dx_dot_6 / dx_1, ... , dx_dot_6 / dx_6, dx_dot_6 / du_1, ..., , dx_dot_6 / du_4 ]
     *
     * or more precisely:

        [ 0, 0, 0, 1, 0, 0,                                                          0,                                                                 0,                                       0,                                                                 0]
        [ 0, 0, 0, 0, 1, 0,                                                          0,                                                                 0,                                       0,                                                                 0]
        [ 0, 0, 0, 0, 0, 1,                                                          0,                                                                 0,                                       0,                                                                 0]
        [ 0, 0, 0, 0, 0, 0, - sin(u_1__)*sin(u_3__) - cos(u_1__)*cos(u_3__)*sin(u_2__), -u_0__*(cos(u_1__)*sin(u_3__) - cos(u_3__)*sin(u_1__)*sin(u_2__)), -u_0__*cos(u_1__)*cos(u_2__)*cos(u_3__), -u_0__*(cos(u_3__)*sin(u_1__) - cos(u_1__)*sin(u_2__)*sin(u_3__))]
        [ 0, 0, 0, 0, 0, 0,   cos(u_3__)*sin(u_1__) - cos(u_1__)*sin(u_2__)*sin(u_3__),  u_0__*(cos(u_1__)*cos(u_3__) + sin(u_1__)*sin(u_2__)*sin(u_3__)), -u_0__*cos(u_1__)*cos(u_2__)*sin(u_3__), -u_0__*(sin(u_1__)*sin(u_3__) + cos(u_1__)*cos(u_3__)*sin(u_2__))]
        [ 0, 0, 0, 0, 0, 0,                                     -cos(u_1__)*cos(u_2__),                                       u_0__*cos(u_2__)*sin(u_1__),             u_0__*cos(u_1__)*sin(u_2__),                                                                 0]

     However, since we will be comparing these to
     */
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
        T *dg_dx_lagrange = dgdx + 14 * n_c * n_w;
        Eigen::Ref<Eigen::Matrix<T, 1, n_w>> times = Get::times(x);
        for (Index i_w = 0; i_w < n_w; ++i_w) {

            /* First, we put in the terms associated with the dynamics */
            Eigen::Ref<Eigen::Matrix<T, n_u, n_c>> u = Get::controlsAtWaypoint(x, i_w);
            auto thrust = u.row(0).array();
            auto phi = u.row(1).array();
            auto theta = u.row(2).array();
            auto psi = u.row(3).array();

            Row sin_phi = sin(phi);
            Row cos_phi = cos(phi);
            Row sin_psi = sin(psi);
            Row cos_psi = cos(psi);
            Row sin_theta = sin(theta);
            Row cos_theta = cos(theta);

            JacEntry(dgdx + 0 * n_c) = 1;
            JacEntry(dgdx + 1 * n_c) = 1;
            JacEntry(dgdx + 2 * n_c) = 1;
            JacEntry(dgdx + 3 * n_c) = -sin_phi * sin_psi - cos_phi * cos_psi * sin_theta;
            JacEntry(dgdx + 4 * n_c) = cos_psi * sin_phi - cos_phi * sin_theta * sin_psi;
            JacEntry(dgdx + 5 * n_c) = -cos_phi * cos_theta;
            JacEntry(dgdx + 6 * n_c) = -thrust * (cos_phi * sin_psi - cos_psi * sin_phi * sin_theta);
            JacEntry(dgdx + 7 * n_c) = thrust * (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi);
            JacEntry(dgdx + 8 * n_c) = thrust * cos_theta * sin_phi;
            JacEntry(dgdx + 9 * n_c) = -thrust * cos_phi * cos_theta * cos_psi;
            JacEntry(dgdx + 10 * n_c) = -thrust * cos_phi * cos_theta * sin_psi;
            JacEntry(dgdx + 11 * n_c) = thrust * cos_phi * sin_theta;
            JacEntry(dgdx + 12 * n_c) = -thrust * (cos_psi * sin_phi - cos_phi * sin_theta * sin_psi);
            JacEntry(dgdx + 13 * n_c) = -thrust * (sin_phi * sin_psi + cos_phi * cos_psi * sin_theta);
            dgdx += 14 * n_c;

            /* Next, we put in the terms associated with the interpolation polynomial.
             * This involves copying the derivative coefficients n_x times for each waypoint. */
            T t_w = times(i_w);
            Eigen::Map<Eigen::Matrix<T, n_c * n_c * n_x, 1>> dG_dX(dg_dx_lagrange);
            dG_dX = negative_derivative_coefficients.template replicate<n_x, 1>() / t_w;
            dg_dx_lagrange += n_c * n_c * n_x;

            /* Finally, the terms due to the time derivatives of the Lagrange interpolation polynomial */
            Map dG_dTime(dg_dx_lagrange);
            dG_dTime = Get::statesAtWaypoint(dx, i_w) / (t_w * t_w);
            dg_dx_lagrange += n_c * n_x;
        }
        return derivatives_in_data;
    }

    /** Return the nonzero rows and columns in the jacobian corresponding to the specified waypoint.
     * It is vitally important that these are returned in the same order as in "evaluateJacobian".
     *
     * First, let's talk about the rows...
     * At each waypoint, the dynamics generate n_x * n_c constraints. In fact, the
     * statesAtWaypoint getter retrieves an n_x x n_c block of states (or state derivatives,
     * depending on the input pointer). That matrix looks like
     *
     * [ x_1(c_1), x_2(c_2), ... x_1(c_{n_c}) ]
     * [                     ...              ]
     * [ x_6(c_1), x_6(c_2), ... x_6(c_{n_c}) ]
     *
     * Since Eigen handles things in column-major order, these constraints
     * are put into storage like
     *
     * [ x_1(c_1), ..., x_6(c_1), x_1(c_2), ..., x_6(c_2), ..., x_1(c_{n_c}), ..., x_6(c_{n_c}) ]
     *
     * In the dynamicsJacobian section comments, we show what the jacobian would look like
     * for a specific waypoint and collocation point. So in practice, for each of the 14
     * nonzero terms listed there, the hard part is to figure out the correct row starting index.
     * Then we use a stride of n_x to place the rest of the terms in the jacobian.
     *
     * Next, as for the columns... We should try not to infer any indices here.
     * That is because we should not be relying on the underlying representation in memory.
     * Otherwise, our implementation would be tied to one specific memory layout.
     * Instead, we will replace our x with a list of indices, and then using that array,
     * extract the indices we need.
     */
    void generateSparsityStructure(T *x,
                                   Bool new_x,
                                   Data *data) override {

        /* We need the right derivative coefficients so that we can efficiently construct the jacobian */
        PD *pd = static_cast<PD *>(data);
        Eigen::Matrix<T, n_c, n_c> derivative_coefficients = pd->getter.rightDerivativeCoefficients().transpose();
        negative_derivative_coefficients = -Eigen::Map<Eigen::Matrix<T, n_c * n_c, 1 >>(derivative_coefficients.data());

        /* Make sure that the number of nonzero rows and columns are properly sized */
        this->nonzero_rows.resize(14 * n_c * n_w + n_x * n_c * n_c * n_w + n_constraints);
        this->nonzero_cols.resize(14 * n_c * n_w + n_x * n_c * n_c * n_w + n_constraints);
        Index *rows = this->nonzero_rows.data();
        Index *cols = this->nonzero_cols.data();

        /* To get the nonzero columns, we essentially replace
         * x with a list of indices in the constraint function, and then
         * see which values are pulled off. */
        Eigen::Matrix<Index, Get::n_vars, 1> inds;
        for (Index i = 0; i < Get::n_vars; ++i)
            inds[i] = i;

        /* For each waypoint ... */
        for (Index i_w = 0; i_w < n_w; ++i_w) {

            /* First we generate a list of the nonzero entries associated with the dynamics */
            Eigen::Ref<Eigen::Matrix<Index, n_x, n_c>> state_indices = GetIndex::statesAtWaypoint(inds.data(), i_w);
            auto vx_indices = state_indices.row(3).array();
            auto vy_indices = state_indices.row(4).array();
            auto vz_indices = state_indices.row(5).array();

            Eigen::Ref<Eigen::Matrix<Index, n_u, n_c>> control_indices = GetIndex::controlsAtWaypoint(inds.data(), i_w);
            auto thrust_indices = control_indices.row(0).array();
            auto phi_indices = control_indices.row(1).array();
            auto theta_indices = control_indices.row(2).array();
            auto psi_indices = control_indices.row(3).array();

            const Index row_offset = i_w * n_x * n_c;
            for (Index i = 0; i < n_c; ++i) {

                /* First the identity terms */
                rows[0 * n_c + i] = row_offset + 0 + i * n_x;
                cols[0 * n_c + i] = vx_indices(i);
                rows[1 * n_c + i] = row_offset + 1 + i * n_x;
                cols[1 * n_c + i] = vy_indices(i);
                rows[2 * n_c + i] = row_offset + 2 + i * n_x;
                cols[2 * n_c + i] = vz_indices(i);

                /* Now the derivatives with respect to the input. */
                rows[3 * n_c + i] = row_offset + 3 + i * n_x;
                cols[3 * n_c + i] = thrust_indices(i);
                rows[4 * n_c + i] = row_offset + 4 + i * n_x;
                cols[4 * n_c + i] = thrust_indices(i);
                rows[5 * n_c + i] = row_offset + 5 + i * n_x;
                cols[5 * n_c + i] = thrust_indices(i);

                rows[6 * n_c + i] = row_offset + 3 + i * n_x;
                cols[6 * n_c + i] = phi_indices(i);
                rows[7 * n_c + i] = row_offset + 4 + i * n_x;
                cols[7 * n_c + i] = phi_indices(i);
                rows[8 * n_c + i] = row_offset + 5 + i * n_x;
                cols[8 * n_c + i] = phi_indices(i);

                rows[9 * n_c + i] = row_offset + 3 + i * n_x;
                cols[9 * n_c + i] = theta_indices(i);
                rows[10 * n_c + i] = row_offset + 4 + i * n_x;
                cols[10 * n_c + i] = theta_indices(i);
                rows[11 * n_c + i] = row_offset + 5 + i * n_x;
                cols[11 * n_c + i] = theta_indices(i);

                rows[12 * n_c + i] = row_offset + 3 + i * n_x;
                cols[12 * n_c + i] = psi_indices(i);
                rows[13 * n_c + i] = row_offset + 4 + i * n_x;
                cols[13 * n_c + i] = psi_indices(i);
            }
            rows += 14 * n_c;
            cols += 14 * n_c;
        }

        /* Now the terms related to the Lagrange interpolation polynomials. */
        Eigen::Matrix<Index, n_c, 1> count;
        for (Index i_c = 0; i_c < n_c; ++i_c)
            count(i_c) = i_c * n_x;
        Eigen::Matrix<Index, n_c * n_c, 1> row_count = count.template replicate<n_c, 1>();

        Eigen::Matrix<Index, n_x * n_c, 1> count_nxc;
        for (Index i = 0; i < n_x * n_c; ++i)
            count_nxc(i) = i;

        Eigen::Matrix<Index, n_c * n_c, 1> col_count;
        for (Index i_c = 0; i_c < n_c; ++i_c)
            col_count.template middleRows<n_c>(i_c * n_c).fill(i_c * (n_x + n_u));

        for (Index i_w = 0; i_w < n_w; ++i_w) {
            for (Index i_x = 0; i_x < n_x; ++i_x) {
                Eigen::Map<Eigen::Matrix<Index, n_c * n_c, 1>>(rows).array() =
                        row_count.array() + i_x + i_w * n_x * n_c;
                rows += n_c * n_c;

                // TODO This portion with the columns is not portable.
                // It should be accomplished using indexing.
                Eigen::Map<Eigen::Matrix<Index, n_c * n_c, 1>>(cols).array() =
                        col_count.array() + i_x + i_w * (n_x + n_u) * n_c;
                cols += n_c * n_c;
            }

            /* The derivative of the Lagrange interpolation polynomial with respect to time. */
            Eigen::Map<Eigen::Matrix<Index, n_x * n_c, 1>> timeRows(rows);
            timeRows.array() = count_nxc.array() + i_w * n_x * n_c;
            rows += n_x * n_c;

            Eigen::Map<Eigen::Matrix<Index, n_x * n_c, 1>> timeCols(cols);
            timeCols.fill(GetIndex::times(inds.data())(i_w));
            cols += n_x * n_c;
        }
    }
};

#endif /* DYNAMICS_CONSTRAINTS_HEADER */