pub mod algebra;
pub mod qdldl;
mod structs;
mod cones;

mod kktsolvers;
mod types;
use num_traits::Float;
use crate::algebra::{MatrixMath, ScalarMath, MatrixVectorMultiply, ShapedMatrix, SymMatrixVectorMultiply, VectorMath};
use clarabel::solver::{DefaultSettingsBuilder, SolverStatus};
use clarabel::solver::traits::Settings;
use crate::algebra::CscMatrix;
use crate::cones::{CompositeCone, Cone};
use crate::cones::SupportedConeT::NonnegativeConeT;
use crate::kktsolvers::direct::DirectLDLKKTSolver;
use crate::kktsolvers::KKTSolver;
use std::io::Write;
use crate::types::*;
// convert a string in LowerExp display format into one that
// 1) always has a sign after the exponent, and
// 2) has at least two digits in the exponent.
// This matches the Julia output formatting.

fn _exp_str_reformat(mut thestr: String) -> String {
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let eidx = thestr.find('e').unwrap();
    let has_sign = thestr.chars().nth(eidx + 1).unwrap() == '-';

    let has_short_exp = {
        if !has_sign {
            thestr.len() == eidx + 2
        } else {
            thestr.len() == eidx + 3
        }
    };

    let chars;
    if !has_sign {
        if has_short_exp {
            chars = "+0";
        } else {
            chars = "+";
        }
    } else if has_short_exp {
        chars = "0";
    } else {
        chars = "";
    }

    let shift = if has_sign { 2 } else { 1 };
    thestr.insert_str(eidx + shift, chars);
    thestr
}


macro_rules! expformat {
    ($fmt:expr,$val:expr) => {
        if $val.is_finite() {
            _exp_str_reformat(format!($fmt, $val))
        } else {
            format!($fmt, $val)
        }
    };
}
fn main() -> Result<(), Box<dyn std::error::Error>> {

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

    let mut cones = CompositeCone::<f64>::new(&[NonnegativeConeT(4)]);

    // some caution is required to ensure we take a minimal,
    // but nonzero, number of data copies during presolve steps


    if !P.is_triu() {
        P = P.to_triu();
    }


    //cap entries in b at INFINITY.  This is important
    //for inf values that were not in a reduced cone
    let infbound = clarabel::solver::get_infinity();
    b.scalarop(|x| f64::min(x, infbound));

    // this ensures m is the *reduced* size m
    let (m, n) = A.size();


    let normq = q.norm_inf();
    let normb = b.norm_inf();

    // -------------------------------------------------------------------


    // ------------------------ Variables --------------------------------

    let mut x = vec![0.0; n];
    let mut s = vec![0.0; m];
    let mut z = vec![0.0; m];
    let mut τ = 1.0;
    let mut κ = 1.0;

    let mut prev_x = vec![0.0; n];
    let mut prev_s = vec![0.0; m];
    let mut prev_z = vec![0.0; m];
    let mut prev_τ = 1.0;
    let mut prev_κ = 1.0;

    let mut step_rhs_x = vec![0.0; n];
    let mut step_rhs_s = vec![0.0; m];
    let mut step_rhs_z = vec![0.0; m];
    let mut step_rhs_τ = 1.0;
    let mut step_rhs_κ = 1.0;


    let mut step_lhs_x = vec![0.0; n];
    let mut step_lhs_s = vec![0.0; m];
    let mut step_lhs_z = vec![0.0; m];
    let mut step_lhs_τ = 1.0;
    let mut step_lhs_κ = 1.0;

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

    // ------------------------ Settings -----------------------------------

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    // -----------------------------------------------------------------


    // ----------------------- KKTSystem --------------------------------

    //the LHS constant part of the reduced solve
    let mut x1 = vec![0.0; n];
    let mut z1 = vec![0.0; m];

    //the LHS for other solves
    let mut x2 = vec![0.0; n];
    let mut z2 = vec![0.0; m];

    //workspace compatible with (x,z)
    let mut workx = vec![0.0; n];
    let mut workz = vec![0.0; m];

    //additional conic workspace vector compatible with s and z
    let mut work_conic = vec![0.0; m];

    let mut kktsolver = DirectLDLKKTSolver::<f64>::new(
        &P,
        &A,
        &cones,
        m,
        n,
        settings.core(),
    );

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

    let mut out = std::io::stdout();

    //print a subheader for the iterations info
    write!(out, "iter    ")?;
    write!(out, "pcost        ")?;
    write!(out, "dcost       ")?;
    write!(out, "gap       ")?;
    write!(out, "pres      ")?;
    write!(out, "dres      ")?;
    write!(out, "k/t       ")?;
    write!(out, " μ       ")?;
    write!(out, "step      ")?;
    writeln!(out,)?;
    writeln!(out,
             "---------------------------------------------------------------------------------------------"
    )?;
    std::io::stdout().flush()?;

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
        μ = (dot_sz + τ * κ) / denom as f64;

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

        let xPx_τinvsq_over2 = dot_xPx * τinv * τinv / 2.0;
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

        // -------------------------------< Info print_status -------------------------------

        if settings.verbose {
            let mut out = std::io::stdout();

            write!(out, "{:>3}  ", iterations)?;
            write!(out, "{}  ", expformat!("{:+8.4e}", cost_primal))?;
            write!(out, "{}  ", expformat!("{:+8.4e}", cost_dual))?;
            let gapprint = f64::min(gap_abs, gap_rel);
            write!(out, "{}  ", expformat!("{:6.2e}", gapprint))?;
            write!(out, "{}  ", expformat!("{:6.2e}", res_primal))?;
            write!(out, "{}  ", expformat!("{:6.2e}", res_dual))?;
            write!(out, "{}  ", expformat!("{:6.2e}", ktratio))?;
            write!(out, "{}  ", expformat!("{:6.2e}", μ))?;

            if iterations > 0 {
                write!(out, "{}  ", expformat!("{:>.2e}", step_length))?;
            } else {
                write!(out, " ------   ")?; //info.step_length
            }

            writeln!(out,)?;

        }



        // -------------------------------</ Info print_status -------------------------------

        // --------------------------- < Info check_termination ------------------

        //  optimality or infeasibility
        // --------------------------- < Info check_convergence_full ------------------
        // "full" tolerances
        let tol_gap_abs = settings.tol_gap_abs;
        let tol_gap_rel = settings.tol_gap_rel;
        let tol_feas = settings.tol_feas;
        let tol_infeas_abs = settings.tol_infeas_abs;
        let tol_infeas_rel = settings.tol_infeas_rel;
        let tol_ktratio = settings.tol_ktratio;

        let solved_status = SolverStatus::Solved;
        let pinf_status = SolverStatus::PrimalInfeasible;
        let dinf_status = SolverStatus::DualInfeasible;

        // --------------------------- < Info check_convergence ------------------

        // --------------------------- < Info is_solved ------------------
        let is_solved = ((gap_abs < tol_gap_abs) || (gap_rel < tol_gap_rel))
            && (res_primal < tol_feas)
            && (res_dual < tol_feas);
        // --------------------------- </ Info is_solved ------------------

        // --------------------------- < Info is_primal_infeasible ------------------
        let is_primal_infeasible = (dot_bz < -tol_infeas_abs)
            && (res_primal_inf < -tol_infeas_rel * dot_bz);
        // --------------------------- </ Info is_primal_infeasible ------------------

        // --------------------------- < Info is_dual_infeasible ------------------
        let is_dual_infeasible = (dot_qx < -tol_infeas_abs)
            && (res_dual_inf < -tol_infeas_rel * dot_qx);
        // --------------------------- </ Info is_dual_infeasible ------------------

        if ktratio <= 1.0 && is_solved {
            status = solved_status;
        //PJG hardcoded factor 1000 here should be fixed
        } else if ktratio > tol_ktratio.recip() * 1000.0 {
            if is_primal_infeasible {
                status = pinf_status;
            } else if is_dual_infeasible {
                status = dinf_status;
            }
        }
        // --------------------------- </ Info check_convergence ------------------

        // --------------------------- </ Info check_convergence_full ------------------

        //  poor progress
        // ----------------------
        if status == SolverStatus::Unsolved
            && iter > 1u32
            && (res_dual > prev_res_dual || res_primal > prev_res_primal)
        {
            // Poor progress at high tolerance.
            if ktratio < f64::epsilon() * 100.0
                && (prev_gap_abs < settings.tol_gap_abs
                || prev_gap_rel < settings.tol_gap_rel)
            {
                status = SolverStatus::InsufficientProgress;
            }

            // Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
            if (res_dual > settings.tol_feas
                && res_dual > prev_res_dual * 100.0)
                || (res_primal > settings.tol_feas
                && res_primal > prev_res_primal * 100.0)
            {
                status = SolverStatus::InsufficientProgress;
            }
        }

        // time or iteration limits
        // ----------------------
        if status == SolverStatus::Unsolved {
            if settings.max_iter == iterations {
                status = SolverStatus::MaxIterations;
            } else if solve_time > settings.time_limit {
                status = SolverStatus::MaxTime;
            }
        }

        // return TRUE if we settled on a final status
        let is_done = status != SolverStatus::Unsolved;

        // --------------------------- </ Info check_termination ------------------

        // check for termination due to slow progress and update strategy
        if is_done {

            // --------------------------- </ solver strategy_checkpoint_insufficient_progress ------------------

            let output;
            if status != SolverStatus::InsufficientProgress {
                // there is no problem, so nothing to do
                output = StrategyCheckpoint::NoUpdate;
            } else {
                // recover old iterate since "insufficient progress" often
                // involves actual degradation of results
                // TODO
                // self.info
                //     .reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

                // If problem is asymmetric, we can try to continue with the dual-only strategy
                if !cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                    status = SolverStatus::Unsolved;
                    output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
                } else {
                    output = StrategyCheckpoint::Fail;
                }
            }
            // --------------------------- </ solver strategy_checkpoint_insufficient_progress ------------------

            match output {
                StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                StrategyCheckpoint::Update(s) => {scaling = s; continue}
            }
        }  // allows continuation if new strategy provided

        // update the scalings
        // --------------

        // ----------------------------------< Variables scale_cones ----------------

        let is_scaling_success = cones.update_scaling(&s, &z, μ, scaling);
        // ----------------------------------</ Variables scale_cones ---------------

        // check whether variables are interior points

        // ----------------------------------< solver strategy_checkpoint_is_scaling_success --------

        let output = if is_scaling_success {
            StrategyCheckpoint::NoUpdate
        } else {
            status = SolverStatus::NumericalError;
            StrategyCheckpoint::Fail
        };
        // ----------------------------------</ solver strategy_checkpoint_is_scaling_success --------
        match output {
            StrategyCheckpoint::Fail => {break}
            StrategyCheckpoint::NoUpdate => {} // we only expect NoUpdate or Fail here
            StrategyCheckpoint::Update(_) => {unreachable!()}
        }

        //increment counter here because we only count
        //iterations that produce a KKT update
        iter += 1;

        // Update the KKT system and the constant parts of its solution.
        // Keep track of the success of each step that calls KKT
        // --------------
        //PJG: This should be a Result in Rust, but needs changes down
        //into the KKT solvers to do that.
        // -------------------------------< KKTSystem update -----------------------------
        // update the linear solver with new cones
        let mut is_kkt_solve_success = kktsolver.update(&cones, settings.core());
        if is_kkt_solve_success {
            // ------------------------------< KKTSystem solve_constant_rhs -----------------
            workx.axpby(-1.0, &q, 0.0); //workx .= -q
            kktsolver.setrhs(&workx, &b);
            is_kkt_solve_success =
                kktsolver
                    .solve(Some(&mut x2), Some(&mut z2), settings.core());

            // ------------------------------</ KKTSystem solve_constant_rhs -----------------
        }

        // calculate KKT solution for constant terms
        // -------------------------------</ KKTSystem update -----------------------------

        // calculate the affine step
        // --------------

        // ------------------------------< Variables affine_step_rhs ----------------------
        step_rhs_x.copy_from(&rx);
        step_rhs_z.copy_from(&rz);
        cones.affine_ds(&mut step_rhs_s, &s);
        step_rhs_τ = rτ;
        step_rhs_κ = τ * κ;
        // ------------------------------</ Variables affine_step_rhs ----------------------

        // ------------------------------< KKTSystem solve -------------------------------

        let (x1, z1) = (&mut x1, &mut z1);
        let (x2, z2) = (&x2, &z2); //from constant solve, so not mut
        let (workx, workz) = (&mut workx, &mut workz);

        // solve for (x1,z1)
        // -----------
        workx.copy_from(&step_rhs_x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        let Δs_const_term = &mut work_conic;

        Δs_const_term.copy_from(&s);

        workz.waxpby(1.0, Δs_const_term, -1.0, &step_rhs_z);

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        kktsolver.setrhs(workx, workz);
        let is_success = kktsolver.solve(Some(x1), Some(z1), settings.core());
        if is_success {
            // solve for Δτ.
            // -----------
            // Numerator first
            let mut ξ = workx.clone();
            ξ.axpby(f64::recip(τ), &x, 0.0);

            let two = 2.0;
            let tau_num = step_rhs_τ - step_rhs_κ / τ
                + q.dot(x1)
                + b.dot(z1)
                + two * P.quad_form(&ξ, x1);

            // offset ξ for the quadratic form in the denominator
            let ξ_minus_x2 = &mut ξ; //alias to ξ, same as workx
            ξ_minus_x2.axpby(-1.0, x2, 1.0);

            let mut tau_den = κ / τ - q.dot(x2) - b.dot(z2);
            tau_den += P.quad_form(ξ_minus_x2, ξ_minus_x2) - P.quad_form(x2, x2);

            // solve for (Δx,Δz)
            // -----------
            step_lhs_τ = tau_num / tau_den;
            step_lhs_x.waxpby(1.0, x1, step_lhs_τ, x2);
            step_lhs_z.waxpby(1.0, z1, step_lhs_τ, z2);

            // solve for Δs
            // -------------
            //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
            //  cones and Hs = μH(z) for asymmetric cones
            cones.mul_Hs(&mut step_lhs_s, &step_lhs_z, workz);
            step_lhs_s.axpby(-1.0, Δs_const_term, -1.0); // lhs.s = -(lhs.s+Δs_const_term);

            // solve for Δκ
            // --------------
            step_lhs_κ = -(step_rhs_κ + κ * step_lhs_τ) / τ;

        }


        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        is_kkt_solve_success = is_kkt_solve_success && is_success;

        // ------------------------------</ KKTSystem solve -------------------------------

        if is_kkt_solve_success {

            //calculate step length and centering parameter
            // --------------

            // ----------------------------< solver get_step_length ------------------------


            // ----------------------------< Variables calc_step_length ------------------------

            let ατ = {
                if step_lhs_τ < 0.0 {
                    -τ / step_lhs_τ
                } else {
                    f64::max_value()
                }
            };

            let ακ = {
                if step_lhs_κ < 0.0 {
                    -κ / step_lhs_κ
                } else {
                    f64::max_value()
                }
            };

            let new_α = [ατ, ακ, 1.0].minimum();
            let (αz, αs) = cones.step_length(&step_lhs_z, &step_lhs_s, &z, &s, settings.core(), new_α);

            // itself only allows for a single maximum value.
            // To enable split lengths, we need to also pass a
            // tuple of limits to the step_length function of
            // every cone
            let mut new_α = f64::min(αz, αs);

            // only for combined step direction
            // new_α *= settings.core().max_step_fraction;



            // ----------------------------</ Variables calc_step_length ------------------------
            // ----------------------------</ solver get_step_length ------------------------
            σ = f64::powi(1.0 - new_α, 3);

            // make a reduced Mehrotra correction in the first iteration
            // to accommodate badly centred starting points
            let m = if iter > 1 {1.0} else {α};

            // calculate the combined step and length
            // --------------

            // ----------------------------< Variables combined_step_rhs -----------------------

            let dotσμ = σ * μ;

            step_rhs_x.axpby(1.0 - σ, &rx, 0.0); //self.x  = (1 - σ)*rx
            step_rhs_τ = (1.0 - σ) * rτ;
            step_rhs_κ = -dotσμ + m * step_lhs_τ * step_lhs_κ + τ * κ;

            // ds is different for symmetric and asymmetric cones:
            // Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
            // Asymmetric cones: d.s = s + σμ*g(z)

            // we want to scale the Mehotra correction in the symmetric
            // case by M, so just scale step_z by M.  This is an unnecessary
            // vector operation (since it amounts to M*z'*s), but it
            // doesn't happen very often
            if m != 1.0 {
                step_lhs_z.scale(m);
            }

            cones.combined_ds_shift(&mut step_rhs_z, &mut step_lhs_z, &mut step_lhs_s, dotσμ);

            //We are relying on d.s = affine_ds already here
            step_rhs_s.axpby(1.0, &step_rhs_z, 1.0);

            // now we copy the scaled res for rz and d.z is no longer work
            step_rhs_z.axpby(1.0 - σ, &rz, 0.0);

            // ----------------------------</ Variables combined_step_rhs -----------------------

            // ------------------------------< KKTSystem solve -------------------------------

            // let (x1, z1) = (&mut x1, &mut z1);
            // let (x2, z2) = (&x2, &z2); //from constant solve, so not mut
            // let (workx, workz) = (&mut workx, &mut workz);

            // solve for (x1,z1)
            // -----------
            workx.copy_from(&step_rhs_x);

            // compute the vector c in the step equation HₛΔz + Δs = -c,
            // with shortcut in affine case
            let Δs_const_term = &mut work_conic;

            cones.Δs_from_Δz_offset(Δs_const_term, &step_rhs_s, &mut step_lhs_z, &z);

            workz.waxpby(1.0, Δs_const_term, -1.0, &step_rhs_z);

            // ---------------------------------------------------
            // this solves the variable part of reduced KKT system
            kktsolver.setrhs(workx, workz);
            let is_success = kktsolver.solve(Some(x1), Some(z1), settings.core());
            if is_success {
                // solve for Δτ.
                // -----------
                // Numerator first
                let ξ = workx;
                ξ.axpby(f64::recip(τ), &x, 0.0);

                let two = 2.0;
                let tau_num = step_rhs_τ - step_rhs_κ / τ
                    + q.dot(x1)
                    + b.dot(z1)
                    + two * P.quad_form(ξ, x1);

                // offset ξ for the quadratic form in the denominator
                let ξ_minus_x2 = ξ; //alias to ξ, same as workx
                ξ_minus_x2.axpby(-1.0, x2, 1.0);

                let mut tau_den = κ / τ - q.dot(x2) - b.dot(z2);
                tau_den += P.quad_form(ξ_minus_x2, ξ_minus_x2) - P.quad_form(x2, x2);

                // solve for (Δx,Δz)
                // -----------
                step_lhs_τ = tau_num / tau_den;
                step_lhs_x.waxpby(1.0, x1, step_lhs_τ, x2);
                step_lhs_z.waxpby(1.0, z1, step_lhs_τ, z2);

                // solve for Δs
                // -------------
                //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
                //  cones and Hs = μH(z) for asymmetric cones
                cones.mul_Hs(&mut step_lhs_s, &step_lhs_z, workz);
                step_lhs_s.axpby(-1.0, Δs_const_term, -1.0); // lhs.s = -(lhs.s+Δs_const_term);

                // solve for Δκ
                // --------------
                step_lhs_κ = -(step_rhs_κ + κ * step_lhs_τ) / τ;

            }


            // we don't check the validity of anything
            // after the KKT solve, so just return is_success
            // without further validation
            is_kkt_solve_success = is_kkt_solve_success && is_success;

            // ------------------------------</ KKTSystem solve -------------------------------
        }

        // --------------------------- < solver strategy_checkpoint_numerical_error ------------------
        let output;
        // No update if kkt updates successfully
        if is_kkt_solve_success {
            output = StrategyCheckpoint::NoUpdate;
        }
        // If problem is asymmetric, we can try to continue with the dual-only strategy
        else if !cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
            output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
        } else {
            // out of tricks.  Bail out with an error
            status = SolverStatus::NumericalError;
            output = StrategyCheckpoint::Fail;
        }

        // --------------------------- </ solver strategy_checkpoint_numerical_error ------------------

        match output {
            StrategyCheckpoint::NoUpdate => {}
            StrategyCheckpoint::Update(s) => {α = 0.0; scaling = s; continue}
            StrategyCheckpoint::Fail => {α = 0.0; break}
        }

        // ----------------------------< solver get_step_length ------------------------


        // ----------------------------< Variables calc_step_length ------------------------

        let ατ = {
            if step_lhs_τ < 0.0 {
                -τ / step_lhs_τ
            } else {
                f64::max_value()
            }
        };

        let ακ = {
            if step_lhs_κ < 0.0 {
                -κ / step_lhs_κ
            } else {
                f64::max_value()
            }
        };

        let new_α = [ατ, ακ, 1.0].minimum();
        let (αz, αs) = cones.step_length(&step_lhs_z, &step_lhs_s, &z, &s, settings.core(), new_α);

        // itself only allows for a single maximum value.
        // To enable split lengths, we need to also pass a
        // tuple of limits to the step_length function of
        // every cone
        let mut new_α = f64::min(αz, αs);
        new_α *= settings.core().max_step_fraction;



        // ----------------------------</ Variables calc_step_length ------------------------

        // additional barrier function limits for asymmetric cones
        if !cones.is_symmetric()
            && scaling == ScalingStrategy::Dual
        {
            // ---------------------------< solver backtrack_step_to_barrier ------------------
            let step = settings.core().linesearch_backtrack_step;
            let mut new_α_1 = new_α;

            for _ in 0..50 {
                // -----------------------< Variables barrier --------------------------------
                let central_coef = (cones.degree() + 1) as f64;

                let cur_τ = τ + new_α_1 * step_lhs_τ;
                let cur_κ = κ + new_α_1 * step_lhs_κ;

                // compute current μ
                let sz = <[f64] as VectorMath<f64>>::dot_shifted(&z, &s, &step_lhs_z, &step_lhs_s, new_α_1);
                let μ = (sz + cur_τ * cur_κ) / central_coef;

                // barrier terms from gap and scalars
                let mut barrier = central_coef * μ.logsafe() - cur_τ.logsafe() - cur_κ.logsafe();

                // barriers from the cones
                let (z, s) = (&z, &s);
                let (dz, ds) = (&step_lhs_z, &step_lhs_s);

                barrier += cones.compute_barrier(z, s, dz, ds, α);

                // -----------------------</ Variables barrier --------------------------------

                if barrier < 1.0 {
                    new_α =  new_α_1;
                    break
                } else {
                    new_α_1 = step * new_α_1;
                }
            }

            // ---------------------------</ solver backtrack_step_to_barrier ------------------
        }
        α = new_α;

        // ----------------------------</ solver get_step_length ------------------------

        // --------------------------- < solver strategy_checkpoint_small_step -------------------
        let output;

        if !cones.is_symmetric()
            && scaling == ScalingStrategy::PrimalDual
            && α < settings.min_switch_step_length
        {
            output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
        } else if α <= f64::max(0.0, settings.min_terminate_step_length) {
            status = SolverStatus::InsufficientProgress;
            output = StrategyCheckpoint::Fail;
        } else {
            output = StrategyCheckpoint::NoUpdate;
        }

        // --------------------------- </ solver strategy_checkpoint_small_step ------------------

        match output {
            StrategyCheckpoint::NoUpdate => {}
            StrategyCheckpoint::Update(s) => {α = 0.0; scaling = s; continue}
            StrategyCheckpoint::Fail => {α = 0.0; break}
        }

        // ----------------------------< Info save_prev_iterate --------------------------------

        prev_cost_primal = cost_primal;
        prev_cost_dual = cost_dual;
        prev_res_primal = res_primal;
        prev_res_dual = res_dual;
        prev_gap_abs = gap_abs;
        prev_gap_rel = gap_rel;


        x.copy_from(&prev_x);
        s.copy_from(&prev_s);
        z.copy_from(&prev_z);
        τ = prev_τ;
        κ = prev_κ;
        // ----------------------------</ Info save_prev_iterate --------------------------------


        // ----------------------------< Variables add_step --------------------------------

        x.axpby(α, &step_lhs_x, 1.0);
        s.axpby(α, &step_lhs_s, 1.0);
        z.axpby(α, &step_lhs_z, 1.0);
        τ += α * step_lhs_τ;
        κ += α * step_lhs_κ;

        // ----------------------------</ Variables add_step --------------------------------
    }

    // Check we if actually took a final step.  If not, we need
    // to recapture the scalars and print one last line
    if α == 0.0 {

        // --------------------------------< Info save_scalars ------------------------------
        μ_info = μ;
        step_length = α;
        sigma = σ;
        iterations = iter;
        // -------------------------------</ Info save_scalars -------------------------------
        // -------------------------------< Info print_status -------------------------------

        if settings.verbose {
            let mut out = std::io::stdout();

            write!(out, "{:>3}  ", iterations)?;
            write!(out, "{}  ", expformat!("{:+8.4e}", cost_primal))?;
            write!(out, "{}  ", expformat!("{:+8.4e}", cost_dual))?;
            let gapprint = f64::min(gap_abs, gap_rel);
            write!(out, "{}  ", expformat!("{:6.2e}", gapprint))?;
            write!(out, "{}  ", expformat!("{:6.2e}", res_primal))?;
            write!(out, "{}  ", expformat!("{:6.2e}", res_dual))?;
            write!(out, "{}  ", expformat!("{:6.2e}", ktratio))?;
            write!(out, "{}  ", expformat!("{:6.2e}", μ))?;

            if iterations > 0 {
                write!(out, "{}  ", expformat!("{:>.2e}", step_length))?;
            } else {
                write!(out, " ------   ")?; //info.step_length
            }

            writeln!(out,)?;

        }



        // -------------------------------</ Info print_status -------------------------------
    }

    Ok(())
}