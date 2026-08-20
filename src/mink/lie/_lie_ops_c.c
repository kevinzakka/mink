/*
 * Fused lie group operations for performance-critical paths.
 *
 * All functions operate on raw C double arrays, avoiding any intermediate
 * Python object creation. The quaternion convention is (w, x, y, z).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* ========================================================================= */
/* Inline math helpers (equivalent to mju_* functions)                       */
/* ========================================================================= */

/* Squared-angle cutoff for cancellation-prone Jacobian coefficients. */
static const double JACOBIAN_TAYLOR_THRESHOLD_SQ = 1e-2;

static inline double vec3_dot(const double *a, const double *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline double vec3_norm(const double *v) {
    return sqrt(vec3_dot(v, v));
}

/* Normalize in place, return original norm. */
static inline double vec3_normalize(double *v) {
    double n = vec3_norm(v);
    if (n > 0.0) {
        double inv = 1.0 / n;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
    return n;
}

/* out = -q (quaternion conjugate for unit quaternions). */
static inline void quat_neg(double *out, const double *q) {
    out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
}

/* out = a * b (Hamilton product). */
static inline void quat_mul(double *out, const double *a, const double *b) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

/* out = rotate v by quaternion q: out = q * [0,v] * q^{-1}. */
static inline void quat_rotate(double *out, const double *v, const double *q) {
    /* Using the formula: out = v + 2*w*(w x v) + 2*(u x (u x v))
       where q = [w, u]. Expanded: */
    double w = q[0], x = q[1], y = q[2], z = q[3];
    /* t = 2 * (u x v) */
    double tx = 2.0 * (y*v[2] - z*v[1]);
    double ty = 2.0 * (z*v[0] - x*v[2]);
    double tz = 2.0 * (x*v[1] - y*v[0]);
    out[0] = v[0] + w*tx + (y*tz - z*ty);
    out[1] = v[1] + w*ty + (z*tx - x*tz);
    out[2] = v[2] + w*tz + (x*ty - y*tx);
}

/* 3x3 skew-symmetric matrix from vector, stored row-major. */
static inline void skew3(double *out, const double *v) {
    out[0] = 0.0;    out[1] = -v[2];  out[2] = v[1];
    out[3] = v[2];   out[4] = 0.0;    out[5] = -v[0];
    out[6] = -v[1];  out[7] = v[0];   out[8] = 0.0;
}

/* C = A @ B for 3x3 row-major matrices. */
static inline void mat33_mul(double *C, const double *A, const double *B) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[3*i+j] = A[3*i+0]*B[0*3+j] + A[3*i+1]*B[1*3+j] + A[3*i+2]*B[2*3+j];
        }
    }
}

/* C = A @ B for 6x6 row-major matrices. */
static void mat66_mul(double *C, const double *A, const double *B) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            double s = 0.0;
            for (int k = 0; k < 6; k++) s += A[6*i+k] * B[6*k+j];
            C[6*i+j] = s;
        }
    }
}

/* M = s * (A @ blockdiag(R^T, R^T)) for 6x6 row-major A and 3x3 row-major R. */
static void mat66_mul_blockdiag_transpose(double *M, const double *A,
                                          const double *R, double s) {
    for (int i = 0; i < 6; i++) {
        for (int b = 0; b < 6; b += 3) {
            const double *row = A + 6*i + b;
            for (int j = 0; j < 3; j++) {
                M[6*i + b + j] =
                    s * (row[0]*R[3*j] + row[1]*R[3*j+1] + row[2]*R[3*j+2]);
            }
        }
    }
}

/* out = in^T for 3x3 row-major matrices. */
static inline void mat33_transpose(double *out, const double *in) {
    out[0] = in[0]; out[1] = in[3]; out[2] = in[6];
    out[3] = in[1]; out[4] = in[4]; out[5] = in[7];
    out[6] = in[2]; out[7] = in[5]; out[8] = in[8];
}

/* ========================================================================= */
/* SO3 helpers                                                               */
/* ========================================================================= */

/* SO3 log: quaternion (w,x,y,z) -> omega (3-vector). */
static void so3_log(double *omega, const double *wxyz) {
    double q[4] = {wxyz[0], wxyz[1], wxyz[2], wxyz[3]};
    /* Ensure w >= 0. */
    if (q[0] < 0.0) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
    double v[3] = {q[1], q[2], q[3]};
    double norm = vec3_normalize(v);
    if (norm == 0.0) {
        omega[0] = 0.0; omega[1] = 0.0; omega[2] = 0.0;
        return;
    }
    double angle = 2.0 * atan2(norm, q[0]);
    omega[0] = angle * v[0];
    omega[1] = angle * v[1];
    omega[2] = angle * v[2];
}

/* SO3 left Jacobian inverse: omega (3-vector) -> 3x3 matrix (row-major). */
static void so3_ljacinv(double *out, const double *omega) {
    double theta = vec3_norm(omega);
    double t2 = theta * theta;
    double beta;
    if (t2 < JACOBIAN_TAYLOR_THRESHOLD_SQ) {
        beta = (1.0/12.0) * (1.0 + t2/60.0 * (1.0 + t2/42.0 * (1.0 + t2/40.0)));
    } else {
        double half = 0.5 * theta;
        beta = (1.0 - half / tan(half)) / t2;
    }
    /* out = I + beta * (outer(omega, omega) - dot(omega, omega) * I) - 0.5 * skew(omega) */
    double inner = vec3_dot(omega, omega);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[3*i+j] = beta * (omega[i]*omega[j] - (i == j ? inner : 0.0));
        }
    }
    /* - 0.5 * skew(omega) */
    double a[3] = {-0.5*omega[0], -0.5*omega[1], -0.5*omega[2]};
    out[0*3+1] += -a[2]; out[0*3+2] += a[1];
    out[1*3+0] += a[2];  out[1*3+2] += -a[0];
    out[2*3+0] += -a[1]; out[2*3+1] += a[0];
    /* + I */
    out[0] += 1.0; out[4] += 1.0; out[8] += 1.0;
}

/* Convert row-major 3x3 rotation matrix to quaternion (w,x,y,z). */
/* Shepperd's method, same algorithm as mju_mat2Quat. */
static void mat33_to_quat(double *wxyz, const double *R) {
    double tr = R[0] + R[4] + R[8];
    if (tr > 0.0) {
        double s = sqrt(tr + 1.0) * 2.0;  /* s = 4*w */
        wxyz[0] = 0.25 * s;
        wxyz[1] = (R[7] - R[5]) / s;
        wxyz[2] = (R[2] - R[6]) / s;
        wxyz[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        double s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;  /* s = 4*x */
        wxyz[0] = (R[7] - R[5]) / s;
        wxyz[1] = 0.25 * s;
        wxyz[2] = (R[1] + R[3]) / s;
        wxyz[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        double s = sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;  /* s = 4*y */
        wxyz[0] = (R[2] - R[6]) / s;
        wxyz[1] = (R[1] + R[3]) / s;
        wxyz[2] = 0.25 * s;
        wxyz[3] = (R[5] + R[7]) / s;
    } else {
        double s = sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;  /* s = 4*z */
        wxyz[0] = (R[3] - R[1]) / s;
        wxyz[1] = (R[2] + R[6]) / s;
        wxyz[2] = (R[5] + R[7]) / s;
        wxyz[3] = 0.25 * s;
    }
    /* Normalize to ensure unit quaternion. */
    double n = sqrt(wxyz[0]*wxyz[0] + wxyz[1]*wxyz[1] +
                    wxyz[2]*wxyz[2] + wxyz[3]*wxyz[3]);
    if (n > 0.0) {
        double inv = 1.0 / n;
        wxyz[0] *= inv; wxyz[1] *= inv; wxyz[2] *= inv; wxyz[3] *= inv;
    }
}

/* ========================================================================= */
/* SE3 core operations                                                       */
/* ========================================================================= */

/*
 * SE3 inverse: wxyz_xyz[7] -> wxyz_xyz[7].
 */
static void _se3_inverse(double *out, const double *a) {
    quat_neg(out, a);
    double neg_t[3] = {-a[4], -a[5], -a[6]};
    quat_rotate(out + 4, neg_t, out);
}

/*
 * SE3 log: wxyz_xyz[7] -> tangent[6].
 * tangent = [v; omega] where v = Vinv @ translation, omega = so3_log(quat).
 */
static void _se3_log(double *tangent, const double *wxyz_xyz) {
    double omega[3];
    so3_log(omega, wxyz_xyz);

    /* The SE3 translation inverse is the SO3 left Jacobian inverse. */
    double vinv[9];
    so3_ljacinv(vinv, omega);

    /* v = vinv @ translation */
    const double *t = wxyz_xyz + 4;
    tangent[0] = vinv[0]*t[0] + vinv[1]*t[1] + vinv[2]*t[2];
    tangent[1] = vinv[3]*t[0] + vinv[4]*t[1] + vinv[5]*t[2];
    tangent[2] = vinv[6]*t[0] + vinv[7]*t[1] + vinv[8]*t[2];
    tangent[3] = omega[0];
    tangent[4] = omega[1];
    tangent[5] = omega[2];
}

/*
 * SE3 inverse multiply: compute (a^{-1} @ b) as wxyz_xyz[7].
 */
static void _se3_inverse_multiply(double *out, const double *a, const double *b) {
    /* a.inverse(): negate quaternion, rotate negated translation. */
    double inv_q[4];
    quat_neg(inv_q, a);
    double neg_t[3] = {-a[4], -a[5], -a[6]};
    double inv_t[3];
    quat_rotate(inv_t, neg_t, inv_q);
    /* multiply: (inv_q, inv_t) @ (b_q, b_t). */
    quat_mul(out, inv_q, b);
    double rotated[3];
    quat_rotate(rotated, b + 4, inv_q);
    out[4] = rotated[0] + inv_t[0];
    out[5] = rotated[1] + inv_t[1];
    out[6] = rotated[2] + inv_t[2];
}

/*
 * SE3 rminus: a.rminus(b) = (b^{-1} @ a).log()  -> tangent[6].
 */
static void _se3_rminus(double *tangent, const double *a, const double *b) {
    double temp[7];
    _se3_inverse_multiply(temp, b, a);
    _se3_log(tangent, temp);
}

/*
 * Q matrix for SE3 Jacobian (Eqn 180 in Sola et al.).
 * c is a 6-vector [v; omega], result is 3x3 row-major.
 */
static void _getQ(double *Q, const double *c) {
    const double *v = c;
    const double *w = c + 3;
    double theta = vec3_norm(w);
    double t2 = theta * theta;
    double A = 0.5;
    double B, C, D;

    if (t2 < JACOBIAN_TAYLOR_THRESHOLD_SQ) {
        B = (1.0/6.0) * (1.0 - t2/20.0 * (1.0 - t2/42.0 * (1.0 - t2/72.0)));
        C = -(1.0/24.0) * (1.0 - t2/30.0 * (1.0 - t2/56.0 * (1.0 - t2/90.0)));
        D = (1.0/120.0) * (1.0 - t2/21.0 * (1.0 - t2/48.0 * (1.0 - 2.0*t2/165.0)));
    } else {
        double t4 = t2 * t2;
        double st = sin(theta);
        double ct = cos(theta);
        B = (theta - st) / (t2 * theta);
        C = (1.0 - 0.5*t2 - ct) / t4;
        D = (2.0*theta - 3.0*st + theta*ct) / (2.0 * t4 * theta);
    }

    double V[9], W[9];
    skew3(V, v);
    skew3(W, w);

    double VW[9], WV[9], WVW[9], VWW[9], VWW_T[9];
    double WVWW[9], WWVW[9];
    mat33_mul(VW, V, W);
    mat33_transpose(WV, VW);  /* WV = (VW)^T for skew matrices of 3-vectors */
    mat33_mul(WVW, WV, W);
    mat33_mul(VWW, VW, W);
    mat33_transpose(VWW_T, VWW);
    mat33_mul(WVWW, WVW, W);
    mat33_mul(WWVW, W, WVW);

    /* Q = A*V + B*(WV + VW + WVW) - C*(VWW - VWW^T - 3*WVW) + D*(WVWW + WWVW) */
    for (int i = 0; i < 9; i++) {
        Q[i] = A * V[i]
             + B * (WV[i] + VW[i] + WVW[i])
             - C * (VWW[i] - VWW_T[i] - 3.0*WVW[i])
             + D * (WVWW[i] + WWVW[i]);
    }
}

/*
 * SE3 jlog from a precomputed tangent = log(T): rjacinv(tangent).
 * Returns 6x6 matrix in row-major order.
 */
static void _se3_jlog_from_tangent(double *jlog, const double *tangent) {
    /* rjacinv(tangent) = ljacinv(-tangent) */
    double neg[6] = {-tangent[0], -tangent[1], -tangent[2],
                     -tangent[3], -tangent[4], -tangent[5]};

    for (int i = 0; i < 36; i++) jlog[i] = 0.0;

    double Qmat[9];
    _getQ(Qmat, neg);

    double lji[9];  /* SO3 ljacinv */
    so3_ljacinv(lji, neg + 3);

    /* jlog[:3,:3] = lji */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            jlog[6*i+j] = lji[3*i+j];

    /* jlog[3:,3:] = lji */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            jlog[6*(i+3)+(j+3)] = lji[3*i+j];

    /* jlog[:3,3:] = -lji @ Q @ lji */
    double QJ[9], JQJ[9];
    mat33_mul(QJ, Qmat, lji);
    mat33_mul(JQJ, lji, QJ);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            jlog[6*i+(j+3)] = -JQJ[3*i+j];
}

/*
 * SE3 jlog: compute rjacinv(log(T)) directly from wxyz_xyz[7].
 * Returns 6x6 matrix in row-major order.
 */
static void _se3_jlog(double *jlog, const double *wxyz_xyz) {
    double tangent[6];
    _se3_log(tangent, wxyz_xyz);
    _se3_jlog_from_tangent(jlog, tangent);
}

/* ========================================================================= */
/* SE3 adjoint operations                                                    */
/* ========================================================================= */

/* Convert quaternion (w,x,y,z) to 3x3 rotation matrix (row-major). */
static void quat_to_mat33(double *R, const double *q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double x2 = x*x, y2 = y*y, z2 = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    double wx = w*x, wy = w*y, wz = w*z;
    R[0] = 1.0 - 2.0*(y2 + z2); R[1] = 2.0*(xy - wz);       R[2] = 2.0*(xz + wy);
    R[3] = 2.0*(xy + wz);       R[4] = 1.0 - 2.0*(x2 + z2); R[5] = 2.0*(yz - wx);
    R[6] = 2.0*(xz - wy);       R[7] = 2.0*(yz + wx);        R[8] = 1.0 - 2.0*(x2 + y2);
}

/*
 * SE3 adjoint: wxyz_xyz[7] -> 6x6 matrix (row-major).
 * adjoint = [[R, skew(t) @ R], [0, R]]
 */
static void _se3_adjoint(double *adj, const double *wxyz_xyz) {
    double R[9];
    quat_to_mat33(R, wxyz_xyz);
    const double *t = wxyz_xyz + 4;

    double skw[9], skR[9];
    skew3(skw, t);
    mat33_mul(skR, skw, R);

    for (int i = 0; i < 36; i++) adj[i] = 0.0;

    /* Top-left 3x3: R */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            adj[6*i+j] = R[3*i+j];

    /* Top-right 3x3: skew(t) @ R */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            adj[6*i+(j+3)] = skR[3*i+j];

    /* Bottom-right 3x3: R */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            adj[6*(i+3)+(j+3)] = R[3*i+j];
}

/*
 * Compute the adjoint of T_fw (frame-to-world inverse) from the xmat.
 * xmat is R_wf (3x3 row-major from MuJoCo). T_fw is pure rotation R_wf^T.
 * adjoint(T_fw) = [[R_wf^T, 0], [0, R_wf^T]] since translation is zero.
 * Result: 6x6 matrix (row-major).
 */
static void _se3_rotation_adjoint_from_xmat(double *adj, const double *xmat) {
    double Rt[9];
    mat33_transpose(Rt, xmat);

    for (int i = 0; i < 36; i++) adj[i] = 0.0;

    /* Top-left 3x3: R^T */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            adj[6*i+j] = Rt[3*i+j];

    /* Bottom-right 3x3: R^T */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            adj[6*(i+3)+(j+3)] = Rt[3*i+j];
}

/* ========================================================================= */
/* Fused task kernels                                                        */
/* ========================================================================= */

/*
 * Fused error and Jacobian map for the frame task.
 *
 * Inputs are the target and frame poses in the world frame (wxyz_xyz[7]).
 * With E_inv = target^{-1} @ frame, the inverse pose residual, and R_wf the
 * frame rotation:
 *
 *   error = -log(E_inv) = target (-) frame
 *   M     = -Jlog6(E_inv) @ blockdiag(R_wf^T, R_wf^T)
 *
 * M maps MuJoCo's world-aligned frame Jacobian to the task Jacobian, i.e.
 * J_task = M @ J_world_aligned: the blockdiag factor re-expresses the
 * Jacobian in the local frame and -Jlog6 is the derivative of the error
 * with respect to a local perturbation of the frame.
 */
static void _se3_frame_task_terms(double *error, double *M,
                                  const double *target, const double *frame) {
    double E_inv[7], tau[6], jlog[36], R_wf[9];
    _se3_inverse_multiply(E_inv, target, frame);
    _se3_log(tau, E_inv);
    for (int i = 0; i < 6; i++) error[i] = -tau[i];
    _se3_jlog_from_tangent(jlog, tau);
    quat_to_mat33(R_wf, frame);
    mat66_mul_blockdiag_transpose(M, jlog, R_wf, -1.0);
}

/*
 * Fused error and Jacobian maps for the relative frame task.
 *
 * Inputs are the target pose in the root frame and the world poses of the
 * frame and root (wxyz_xyz[7] each). With T_rf = root^{-1} @ frame the
 * frame pose in the root, E_inv = target^{-1} @ T_rf the inverse pose
 * residual, and R_wf, R_wr the frame and root rotations:
 *
 *   error   = -log(E_inv) = target (-) T_rf
 *   M_frame = -Jlog6(E_inv) @ blockdiag(R_wf^T, R_wf^T)
 *   M_root  =  Jlog6(E_inv) @ Ad(T_rf^{-1}) @ blockdiag(R_wr^T, R_wr^T)
 *
 * This mirrors _se3_frame_task_terms with the world replaced by the root.
 * The task Jacobian is recovered from MuJoCo's world-aligned frame
 * Jacobians as J_task = M_frame @ J_frame + M_root @ J_root.
 */
static void _se3_relative_frame_task_terms(
    double *error, double *M_frame, double *M_root,
    const double *target, const double *frame_world, const double *root_world) {
    double T_rf[7], E_inv[7], tau[6], jlog[36], R[9];
    _se3_inverse_multiply(T_rf, root_world, frame_world);
    _se3_inverse_multiply(E_inv, target, T_rf);
    _se3_log(tau, E_inv);
    for (int i = 0; i < 6; i++) error[i] = -tau[i];
    _se3_jlog_from_tangent(jlog, tau);
    quat_to_mat33(R, frame_world);
    mat66_mul_blockdiag_transpose(M_frame, jlog, R, -1.0);

    double T_fr[7], adj[36], jlog_adj[36];
    _se3_inverse(T_fr, T_rf);
    _se3_adjoint(adj, T_fr);
    mat66_mul(jlog_adj, jlog, adj);
    quat_to_mat33(R, root_world);
    mat66_mul_blockdiag_transpose(M_root, jlog_adj, R, 1.0);
}

/* ========================================================================= */
/* Python wrapper functions                                                  */
/* ========================================================================= */

/* Helper: extract a contiguous double array from a numpy array argument. */
static int parse_array(PyObject *arg, double *buf, npy_intp expected_len) {
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return -1;
    if (PyArray_SIZE(arr) != expected_len) {
        PyErr_Format(PyExc_ValueError,
            "Expected array of length %ld, got %ld",
            (long)expected_len, (long)PyArray_SIZE(arr));
        Py_DECREF(arr);
        return -1;
    }
    double *data = (double *)PyArray_DATA(arr);
    for (npy_intp i = 0; i < expected_len; i++) buf[i] = data[i];
    Py_DECREF(arr);
    return 0;
}

/* Helper: allocate a new 1-D (len d0) or 2-D (d0 x d1) numpy array from buf. */
static PyObject *make_array(const double *buf, int ndim, npy_intp d0, npy_intp d1) {
    npy_intp dims[2] = {d0, d1};
    PyObject *out = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (!out) return NULL;
    double *odata = (double *)PyArray_DATA((PyArrayObject *)out);
    npy_intp size = ndim == 1 ? d0 : d0 * d1;
    for (npy_intp i = 0; i < size; i++) odata[i] = buf[i];
    return out;
}

static PyObject *py_se3_log(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    double wxyz_xyz[7];
    if (parse_array(arg, wxyz_xyz, 7) < 0) return NULL;

    double tangent[6];
    _se3_log(tangent, wxyz_xyz);
    return make_array(tangent, 1, 6, 0);
}

static PyObject *py_se3_inverse_multiply(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg_a, *arg_b;
    if (!PyArg_ParseTuple(args, "OO", &arg_a, &arg_b)) return NULL;
    double a[7], b[7];
    if (parse_array(arg_a, a, 7) < 0) return NULL;
    if (parse_array(arg_b, b, 7) < 0) return NULL;

    double result[7];
    _se3_inverse_multiply(result, a, b);
    return make_array(result, 1, 7, 0);
}

static PyObject *py_se3_rminus(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg_a, *arg_b;
    if (!PyArg_ParseTuple(args, "OO", &arg_a, &arg_b)) return NULL;
    double a[7], b[7];
    if (parse_array(arg_a, a, 7) < 0) return NULL;
    if (parse_array(arg_b, b, 7) < 0) return NULL;

    double tangent[6];
    _se3_rminus(tangent, a, b);
    return make_array(tangent, 1, 6, 0);
}

static PyObject *py_se3_jlog(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    double wxyz_xyz[7];
    if (parse_array(arg, wxyz_xyz, 7) < 0) return NULL;

    double jlog[36];
    _se3_jlog(jlog, wxyz_xyz);
    return make_array(jlog, 2, 6, 6);
}

static PyObject *py_se3_adjoint(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    double wxyz_xyz[7];
    if (parse_array(arg, wxyz_xyz, 7) < 0) return NULL;

    double adj[36];
    _se3_adjoint(adj, wxyz_xyz);
    return make_array(adj, 2, 6, 6);
}

static PyObject *py_se3_rotation_adjoint_from_xmat(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    double xmat[9];
    if (parse_array(arg, xmat, 9) < 0) return NULL;

    double adj[36];
    _se3_rotation_adjoint_from_xmat(adj, xmat);
    return make_array(adj, 2, 6, 6);
}

static PyObject *py_se3_inverse(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    double wxyz_xyz[7];
    if (parse_array(arg, wxyz_xyz, 7) < 0) return NULL;

    double result[7];
    _se3_inverse(result, wxyz_xyz);
    return make_array(result, 1, 7, 0);
}

static PyObject *py_xmat_xpos_to_wxyz_xyz(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg_xmat, *arg_xpos;
    if (!PyArg_ParseTuple(args, "OO", &arg_xmat, &arg_xpos)) return NULL;
    double xmat[9], xpos[3];
    if (parse_array(arg_xmat, xmat, 9) < 0) return NULL;
    if (parse_array(arg_xpos, xpos, 3) < 0) return NULL;

    double result[7];
    mat33_to_quat(result, xmat);
    result[4] = xpos[0];
    result[5] = xpos[1];
    result[6] = xpos[2];
    return make_array(result, 1, 7, 0);
}

static PyObject *py_se3_frame_task_terms(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg_target, *arg_frame;
    if (!PyArg_ParseTuple(args, "OO", &arg_target, &arg_frame)) return NULL;
    double target[7], frame[7];
    if (parse_array(arg_target, target, 7) < 0) return NULL;
    if (parse_array(arg_frame, frame, 7) < 0) return NULL;

    double error[6], M[36];
    _se3_frame_task_terms(error, M, target, frame);

    PyObject *error_out = make_array(error, 1, 6, 0);
    if (!error_out) return NULL;
    PyObject *M_out = make_array(M, 2, 6, 6);
    if (!M_out) { Py_DECREF(error_out); return NULL; }
    return Py_BuildValue("NN", error_out, M_out);
}

static PyObject *py_se3_relative_frame_task_terms(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *arg_target, *arg_frame, *arg_root;
    if (!PyArg_ParseTuple(args, "OOO", &arg_target, &arg_frame, &arg_root)) {
        return NULL;
    }
    double target[7], frame[7], root[7];
    if (parse_array(arg_target, target, 7) < 0) return NULL;
    if (parse_array(arg_frame, frame, 7) < 0) return NULL;
    if (parse_array(arg_root, root, 7) < 0) return NULL;

    double error[6], M_frame[36], M_root[36];
    _se3_relative_frame_task_terms(error, M_frame, M_root, target, frame, root);

    PyObject *error_out = make_array(error, 1, 6, 0);
    if (!error_out) return NULL;
    PyObject *M_frame_out = make_array(M_frame, 2, 6, 6);
    if (!M_frame_out) { Py_DECREF(error_out); return NULL; }
    PyObject *M_root_out = make_array(M_root, 2, 6, 6);
    if (!M_root_out) { Py_DECREF(error_out); Py_DECREF(M_frame_out); return NULL; }
    return Py_BuildValue("NNN", error_out, M_frame_out, M_root_out);
}

/* ========================================================================= */
/* Module definition                                                         */
/* ========================================================================= */

static PyMethodDef methods[] = {
    {"se3_log", py_se3_log, METH_VARARGS,
     "SE3 log map: wxyz_xyz[7] -> tangent[6]"},
    {"se3_inverse_multiply", py_se3_inverse_multiply, METH_VARARGS,
     "Compute (a.inverse() @ b): a[7], b[7] -> result[7]"},
    {"se3_rminus", py_se3_rminus, METH_VARARGS,
     "SE3 rminus: a[7], b[7] -> tangent[6]"},
    {"se3_jlog", py_se3_jlog, METH_VARARGS,
     "SE3 jlog: wxyz_xyz[7] -> 6x6 matrix"},
    {"se3_adjoint", py_se3_adjoint, METH_VARARGS,
     "SE3 adjoint: wxyz_xyz[7] -> 6x6 matrix"},
    {"se3_rotation_adjoint_from_xmat", py_se3_rotation_adjoint_from_xmat, METH_VARARGS,
     "Adjoint of pure rotation from MuJoCo xmat[9] -> 6x6 matrix"},
    {"se3_inverse", py_se3_inverse, METH_VARARGS,
     "SE3 inverse: wxyz_xyz[7] -> wxyz_xyz[7]"},
    {"xmat_xpos_to_wxyz_xyz", py_xmat_xpos_to_wxyz_xyz, METH_VARARGS,
     "Convert MuJoCo xmat[9] + xpos[3] to SE3 wxyz_xyz[7]"},
    {"se3_frame_task_terms", py_se3_frame_task_terms, METH_VARARGS,
     "Frame task error and Jacobian map: target[7], frame[7] -> "
     "(error[6], M[6,6])"},
    {"se3_relative_frame_task_terms", py_se3_relative_frame_task_terms,
     METH_VARARGS,
     "Relative frame task error and Jacobian maps: target[7], frame[7], "
     "root[7] -> (error[6], M_frame[6,6], M_root[6,6])"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_lie_ops_c",
    .m_doc = "Fused lie group operations (C implementation)",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__lie_ops_c(void) {
    import_array();
    return PyModule_Create(&module);
}
