use num_traits::Float;
use clarabel::algebra::{AsFloatT, CscMatrix, MatrixVectorMultiply, ShapedMatrix, SymMatrixVectorMultiply, VectorMath};
use clarabel::solver::core::ScalingStrategy;
use clarabel::solver::{NonnegativeConeT, SolverStatus};

fn main() {

    // -------------------- ProblemData ----------------------------
    // the struct CscMatrix is used throughout the solver,
    // It is an implementation to represent sparse matrices in
    // a compressed sparse column format
    // we can just use it as is as it's very relied upon in the
    // calculations and it saves on memory usage for large problems
    let mut P: CscMatrix<f64> = CscMatrix::<f64>::zeros((2, 2));
    let q = vec![1., -1.];

    // easier way - use the From trait to construct A:
    let A = CscMatrix::from(&[
        [1., 0.],  //
        [0., 1.],  //
        [-1., 0.], //
        [0., -1.], //
    ]);
    let mut b = vec![1.; 4];

    let cones = [NonnegativeConeT(4)];

    // some caution is required to ensure we take a minimal,
    // but nonzero, number of data copies during presolve steps


    if !P.is_triu() {
        P = P.to_triu();
    }


    //cap entries in b at INFINITY.  This is important
    //for inf values that were not in a reduced cone
    let infbound = crate::solver::get_infinity();
    b.scalarop(|x| f64::min(x, infbound));

    // this ensures m is the *reduced* size m
    let (m, n) = A.size();


    let normq = q.norm_inf();
    let normb = b.norm_inf();

    // -------------------------------------------------------------------


    // ------------------------ Variables --------------------------------

    let x = vec![0.0; n];
    let s = vec![0.0; m];
    let z = vec![0.0; m];
    let τ = 1.0;
    let κ = 1.0;

    // ------------------------------------------------------------------

    // ----------------------- Risiduals --------------------------------

    let mut rx = vec![0.0; n];
    let mut rz = vec![0.0; m];
    let mut rτ = 1.0;

    let mut rx_inf = vec![0.0; n];
    let mut rz_inf = vec![0.0; m];

    let mut Px = vec![0.0; n];

    let mut dot_qx = 0.0;
    let mut dot_bz = 0.0;
    let mut dot_sz = 0.0;
    let mut dot_xPx = 0.0;

    // -----------------------------------------------------------------


    // ------------------------ Info -----------------------------------

    let mut μ_info = 0.0;
    let mut sigma = 0.0;
    let mut step_length = 0.0;
    let mut iterations = 0;
    let mut cost_primal = 0.0;
    let mut cost_dual = 0.0;
    let mut res_primal = 0.0;
    let mut res_dual = 0.0;
    let mut res_primal_inf = 0.0;
    let mut res_dual_inf = 0.0;
    let mut gap_abs = 0.0;
    let mut gap_rel = 0.0;
    let mut ktratio = 0.0;

    // previous iterate
    let mut prev_cost_primal = 0.0;
    let mut prev_cost_dual = 0.0;
    let mut prev_res_primal = 0.0;
    let mut prev_res_dual = 0.0;
    let mut prev_gap_abs = 0.0;
    let mut prev_gap_rel = 0.0;

    let mut solve_time = 0.0;
    let mut status = SolverStatus::Unsolved;

    // -----------------------------------------------------------------

    // ----------------------- DefaultEqulibriationData ----------------

    let d = vec![1.0; n];
    let dinv = vec![1.0; n];
    let e = vec![1.0; m];
    let einv = vec![1.0; m];

    let c = 1.0;

    // ----------------------------------------------------------------

    // solver loop
    let mut iter: u32 = 0;
    let mut σ = 1.0;
    let mut α = 0.0;
    let mut μ;


    // still working out how this is used
    let mut scaling = {
        if cones.iter()
            .all(|cone| cone.allows_primal_dual_scaling()) {ScalingStrategy::PrimalDual}
        else {ScalingStrategy::Dual}
    };

    loop {

        // --------------------------- < Risiduals update() ----------------------
        let qx = q.dot(&x);
        let bz = b.dot(&z);
        let sz = s.dot(&z);

        //Px = P*x, P treated as symmetric
        let symP = P.sym();
        symP.symv(&mut Px, &x, 1.0, 0.0);

        let xPx = x.dot(&Px);

        //partial residual calc so we can check primal/dual
        //infeasibility conditions

        //Same as:
        //rx_inf .= -data.A'* variables.z
        let At = A.t();
        At.gemv(&mut rx_inf, &z, -1.0, 0.0);

        //Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
        rz_inf.copy_from(&s);
        let A = &A;
        A.gemv(&mut rz_inf, &x, 1.0, 1.0);

        //complete the residuals
        //rx = rx_inf - Px - qτ
        rx.waxpby(-1.0, &Px, -τ, &q);
        rx.axpby(1.0, &rx_inf, 1.0);

        // rz = rz_inf - bτ
        rz
            .waxpby(1.0, &rz_inf, -τ, &b);

        // τ = qz + bz + κ + xPx/τ;
        rτ = qx + bz + κ + xPx / τ;

        //save local versions
        dot_qx = qx;
        dot_bz = bz;
        dot_sz = sz;
        dot_xPx = xPx;

        // --------------------------- </ Risiduals update() ----------------------

        // --------------------------- < Variables calculate_mu ------------------

        //calculate duality gap (scaled)
        let denom = cones.degree() + 1;
        μ = (dot_sz + τ * κ) / denom;

        // --------------------------- </ Variables calculate_mu -----------------

        // --------------------------- < Info save_scalar ------------------
        μ_info = μ;
        step_length = α;
        sigma = σ;
        iterations = iter;
        // --------------------------- </ Info save_scalar ------------------

        // --------------------------- < Info update ------------------

        // optimality termination check should be computed w.r.t
        // the pre-homogenization x and z variables.
        let τinv = f64::recip(τ);


        // shortcuts for the equilibration matrices
        let dinv = &dinv;
        let einv = &einv;
        let cscale = c;

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = dot_xPx * τinv * τinv / (2.).as_T();
        cost_primal = (dot_qx * τinv + xPx_τinvsq_over2) / cscale;
        cost_dual = (-dot_bz * τinv - xPx_τinvsq_over2) / cscale;

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway
        let mut normx = x.norm_scaled(dinv);
        let mut normz = z.norm_scaled(einv);
        let mut norms = s.norm_scaled(einv);

        // primal and dual infeasibility residuals.
        res_primal_inf = rx_inf.norm_scaled(dinv) / f64::max(1.0, normz);
        res_dual_inf = f64::max(
            Px.norm_scaled(dinv) / f64::max(1.0, normx),
            rz_inf.norm_scaled(einv) / f64::max(1.0, normx + norms),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv;
        normz *= τinv;
        norms *= τinv;

        // primal and dual relative residuals.
        res_primal =
            rz.norm_scaled(einv) * τinv / f64::max(1.0, normb + normx + norms);
        res_dual =
            rx.norm_scaled(dinv) * τinv / f64::max(1.0, normq + normx + normz);

        // absolute and relative gaps
        gap_abs = f64::abs(cost_primal - cost_dual);
        gap_rel = gap_abs
            / f64::max(
            1.0,
            f64::min(f64::abs(cost_primal), f64::abs(cost_dual)),
        );

        // κ/τ
        ktratio = κ / τ;


        // --------------------------- </ Info update ------------------


        // --------------------------- < Info check_termination ------------------


        let is_done = status != SolverStatus::Unsolved;

        // --------------------------- </ Info check_termination ------------------


    }
}