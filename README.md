# clarabel-reimplementation
## main solver struct is in /src/solver/core/solver.rs
```rust
pub struct Solver<D, V, R, K, C, I, SO, SE> {
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub prev_vars: V,
    pub info: I,
    pub solution: SO,
    pub settings: SE,
    pub timers: Option<Timers>,
}
```
## Solver is generic over traits defined in /src/solver/core/traits.rs
### D: a type implementing `ProblemData`
```rust
/// Data for a conic optimization problem.

pub trait ProblemData<T: FloatT> {
    type V: Variables<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Equilibrate internal data before solver starts.
    fn equilibrate(&mut self, cones: &Self::C, settings: &Self::SE);
}

```

### V: a type implementing `Variables`
```rust
/// Variables for a conic optimization problem.

pub trait Variables<T: FloatT> {
    type D: ProblemData<T>;
    type R: Residuals<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Compute the scaled duality gap.

    fn calc_mu(&mut self, residuals: &Self::R, cones: &Self::C) -> T;

    /// Compute the KKT RHS for a pure Newton step.

    fn affine_step_rhs(&mut self, residuals: &Self::R, variables: &Self, cones: &Self::C);

    /// Compute the KKT RHS for an interior point centering step.

    #[allow(clippy::too_many_arguments)]
    fn combined_step_rhs(
        &mut self,
        residuals: &Self::R,
        variables: &Self,
        cones: &mut Self::C,
        step: &mut Self, //mut allows step to double as working space
        σ: T,
        μ: T,
        m: T,
    );

    /// Compute the maximum step length possible in the given
    /// step direction without violating a cone boundary.

    fn calc_step_length(
        &self,
        step_lhs: &Self,
        cones: &mut Self::C,
        settings: &Self::SE,
        step_direction: StepDirection,
    ) -> T;

    /// Update the variables in the given step direction, scaled by `α`.
    fn add_step(&mut self, step_lhs: &Self, α: T);

    /// Bring the variables into the interior of the cone constraints.
    fn symmetric_initialization(&mut self, cones: &mut Self::C);

    /// Initialize all conic variables to unit values.
    fn unit_initialization(&mut self, cones: &Self::C);

    /// Overwrite values with those from another object
    fn copy_from(&mut self, src: &Self);

    /// Apply NT scaling to a collection of cones.

    fn scale_cones(&self, cones: &mut Self::C, μ: T, scaling_strategy: ScalingStrategy) -> bool;

    /// Compute the barrier function

    fn barrier(&self, step: &Self, α: T, cones: &mut Self::C) -> T;

    /// Rescale variables, e.g. to renormalize iterates
    /// in a homogeneous embedding

    fn rescale(&mut self);
}

```

### R: a type implementing `Residuals`
```rust

/// Residuals for a conic optimization problem.

pub trait Residuals<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;

    /// Compute residuals for the given variables.
    ///
    fn update(&mut self, variables: &Self::V, data: &Self::D);
}

```

### K: a type implementing `KKTSystem`
```rust
/// KKT linear solver object.

pub trait KKTSystem<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Update the KKT system.   In particular, update KKT
    /// matrix entries with new variable and refactor.

    fn update(&mut self, data: &Self::D, cones: &Self::C, settings: &Self::SE) -> bool;

    /// Solve the KKT system for the given RHS.

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        step_lhs: &mut Self::V,
        step_rhs: &Self::V,
        data: &Self::D,
        variables: &Self::V,
        cones: &mut Self::C,
        step_direction: StepDirection,
        settings: &Self::SE,
    ) -> bool;

    /// Find an IP starting condition

    fn solve_initial_point(
        &mut self,
        variables: &mut Self::V,
        data: &Self::D,
        settings: &Self::SE,
    ) -> bool;
}

```

### C: a type implementing the trait `Cone` in /src/core/cones/mod.rs
```rust
#[enum_dispatch]
pub trait Cone<T>
where
    T: FloatT,
{
    // functions relating to basic sizing
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;

    //Can the cone provide a sparse expanded representation?
    fn is_sparse_expandable(&self) -> bool;

    // is the cone symmetric?  NB: zero cone still reports true
    fn is_symmetric(&self) -> bool;

    // report false here if only dual scaling is implemented (e.g. GenPowerCone)
    fn allows_primal_dual_scaling(&self) -> bool;

    // converts an elementwise scaling into
    // a scaling that preserves cone memership
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool;

    // returns (α,β) such that:
    // z - α⋅e is just on the cone boundary, with value
    // α >=0 indicates z \in cone, i.e. negative margin ===
    // outside of the cone.
    //
    // β is the sum of the margins that are positive.   For most
    // cones this will just be β = max(0.,α), but for cones that
    // are composites (e.g. the R_n^+), it is the sum of all of
    // the positive margin terms.
    fn margins(&mut self, z: &mut [T], pd: PrimalOrDualCone) -> (T, T);

    // functions relating to unit vectors and cone initialization
    fn scaled_unit_shift(&self, z: &mut [T], α: T, pd: PrimalOrDualCone);
    fn unit_initialization(&self, z: &mut [T], s: &mut [T]);

    // Compute scaling points
    fn set_identity_scaling(&mut self);
    fn update_scaling(
        &mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy
    ) -> bool;

    // operations on the Hessian of the centrality condition
    // : W^TW for symmmetric cones
    // : μH(s) for nonsymmetric cones
    fn Hs_is_diagonal(&self) -> bool;
    fn get_Hs(&self, Hsblock: &mut [T]);
    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]);

    // ---------------------------------------------------------
    // Linearized centrality condition functions
    //
    // For nonsymmetric cones:
    // -----------------------
    //
    // The centrality condition is : s = -μg(z)
    //
    // The linearized version is :
    //     Δs + μH(z)Δz = -ds = -(affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is s
    // The shift term is μg(z) plus any higher order corrections
    //
    // # To recover Δs from Δz, we can write
    //     Δs = - (ds + μHΔz)
    // The "offset" in Δs_from_Δz_offset is then just ds
    //
    // For symmetric cones:
    // --------------------
    //
    // The centrality condition is : (W(z + Δz) ∘ W⁻ᵀ(s + Δs) = μe
    //
    // The linearized version is :
    //     λ ∘ (WΔz + WᵀΔs) = -ds = - (affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is λ ∘ λ
    // The shift term is W⁻¹Δs_aff ∘ WΔz_aff - σμe, where the terms
    // Δs_aff an Δz_aff are from the affine KKT solve, i.e. they
    // are the Mehrotra correction terms.
    //
    // To recover Δs from Δz, we can write
    //     Δs = - ( Wᵀ(λ \ ds) + WᵀW Δz)
    // The "offset" in Δs_from_Δz_offset is then Wᵀ(λ \ ds)
    //
    // Note that the Δs_from_Δz_offset function is only needed in the
    // general combined step direction.   In the affine step direction,
    // we have the identity Wᵀ(λ \ (λ ∘ λ )) = s.  The symmetric and
    // nonsymmetric cases coincide and offset is taken directly as s.
    //
    // The affine step directions terms steps_z and step_s are
    // passed to combined_ds_shift as mutable.  Once they have been
    // used to compute the combined ds shift they are no longer needed,
    // so may be modified in place as workspace.
    // ---------------------------------------------------------
    fn affine_ds(&self, ds: &mut [T], s: &[T]);
    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T);
    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], z: &[T]);

    // Find the maximum step length in some search direction
    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T);

    // return the barrier function at (z+αdz,s+αds)
    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T;
}

```

### I: a type implementing `Info`
```rust
/// Internal information for the solver to monitor progress and check for termination.

pub trait Info<T>: InfoPrint<T>
where
T: FloatT,
{
type V: Variables<T>;
type R: Residuals<T>;

    /// Reset internal data, particularly solve timers.
    fn reset(&mut self, timers: &mut Timers);

    /// Final convergence checks, e.g. for "almost" convergence cases
    fn post_process(&mut self, residuals: &Self::R, settings: &Self::SE);

    /// Compute final values before solver termination
    fn finalize(&mut self, timers: &mut Timers);

    /// Update solver progress information
    fn update(
        &mut self,
        data: &mut Self::D,
        variables: &Self::V,
        residuals: &Self::R,
        timers: &Timers,
    );

    /// Return `true` if termination conditions have been reached.
    fn check_termination(&mut self, residuals: &Self::R, settings: &Self::SE, iter: u32) -> bool;

    // save and recover prior iterates
    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V);
    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V);

    /// Record some of the top level solver's choice of various
    /// scalars. `μ = ` normalized gap.  `α = ` computed step length.
    /// `σ = ` multiplier for the updated centering parameter.
    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32);

    /// Report or update termination status
    fn get_status(&self) -> SolverStatus;
    fn set_status(&mut self, status: SolverStatus);
}
```

### SO: a type implementing `Solution`
```rust
/// Solution for a conic optimization problem.

pub trait Solution<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;
    type I: Info<T>;
    type SE: Settings<T>;

    /// Compute solution from the Variables at solver termination
    fn post_process(
        &mut self,
        data: &Self::D,
        variables: &mut Self::V,
        info: &Self::I,
        settings: &Self::SE,
    );

    /// finalize the solution, e.g. extract final timing from info
    fn finalize(&mut self, info: &Self::I);
}

```

### SE: a type implementing `Settings`
```rust
/// Settings for a conic optimization problem.
///
/// Implementors of this trait can define any internal or problem
/// specific settings they wish.   They must, however, also maintain
/// a settings object of type [`CoreSettings`](crate::solver::core::CoreSettings)
/// and return this to the solver internally.

pub trait Settings<T: FloatT> {
/// Return the core settings.
fn core(&self) -> &CoreSettings<T>;

    /// Return the core settings (mutably).
    fn core_mut(&mut self) -> &mut CoreSettings<T>;
}


```

### Finally The main solver loop with their "novel homogeneous embedding" technique, this function is implemented from the trait `IPSolver`

```rust

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

/// An interior point solver implementing a predictor-corrector scheme

// Only the main solver function lives in IPSolver, since this is the
// only publicly facing trait we want to give the solver.   Additional
// internal functionality for the top level solver object is implemented
// for the IPSolverUtilities trait below, upon which IPSolver depends

pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    /// Run the solver
    fn solve(&mut self);
}

```

The implementation
```rust
    fn solve(&mut self) {
    // various initializations
    let mut iter: u32 = 0;
    let mut σ = T::one();
    let mut α = T::zero();
    let mut μ;

    //timers is stored as an option so that
    //we can swap it out here and avoid
    //borrow conflicts with other fields.
    let mut timers = self.timers.take().unwrap();

    // solver release info, solver config
    // problem dimensions, cone types etc
    notimeit! {timers; {
            _print_banner(self.settings.core().verbose).unwrap();
            self.info.print_configuration(&self.settings, &self.data, &self.cones).unwrap();
            self.info.print_status_header(&self.settings).unwrap();
        }}

    self.info.reset(&mut timers);

    timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            self.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        let mut scaling = {
            if self.cones.allows_primal_dual_scaling() {ScalingStrategy::PrimalDual}
            else {ScalingStrategy::Dual}
        };

        loop {

            //update the residuals
            //--------------
            self.residuals.update(&self.variables, &self.data);

            //calculate duality gap (scaled)
            //--------------
            μ = self.variables.calc_mu(&self.residuals, &self.cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            self.info.save_scalars(μ, α, σ, iter);

            // convergence check and printing
            // --------------
            self.info.update(
                &mut self.data,
                &self.variables,
                &self.residuals,&timers);

            notimeit!{timers; {
                self.info.print_status(&self.settings).unwrap();
            }}

            let isdone = self.info.check_termination(&self.residuals, &self.settings, iter);

            // check for termination due to slow progress and update strategy
            if isdone{
                    match self.strategy_checkpoint_insufficient_progress(scaling){
                        StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                        StrategyCheckpoint::Update(s) => {scaling = s; continue}
                    }
            }  // allows continuation if new strategy provided


            // update the scalings
            // --------------
            let is_scaling_success;
            timeit!{timers => "scale cones"; {
                is_scaling_success = self.variables.scale_cones(&mut self.cones,μ,scaling);
            }}
            // check whether variables are interior points
            match self.strategy_checkpoint_is_scaling_success(is_scaling_success,scaling){
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
            let mut is_kkt_solve_success : bool;
            timeit!{timers => "kkt update"; {
                is_kkt_solve_success = self.kktsystem.update(&self.data, &self.cones, &self.settings);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            self.step_rhs
                .affine_step_rhs(&self.residuals, &self.variables, &self.cones);

            timeit!{timers => "kkt solve"; {
                is_kkt_solve_success = is_kkt_solve_success &&
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Affine,
                    &self.settings,
                );
            }}  //end "kkt solve affine" timer

            // combined step only on affine step success
            if is_kkt_solve_success {

                //calculate step length and centering parameter
                // --------------
                α = self.get_step_length(StepDirection::Affine, scaling);
                σ = self.centering_parameter(α);

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                let m = if iter > 1 {T::one()} else {α};

                // calculate the combined step and length
                // --------------
                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    &mut self.step_lhs,
                    σ,
                    μ,
                    m
                );

                timeit!{timers => "kkt solve" ; {
                    is_kkt_solve_success =
                    self.kktsystem.solve(
                        &mut self.step_lhs,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }} //end "kkt solve"
            }

            // check for numerical failure and update strategy
            match self.strategy_checkpoint_numerical_error(is_kkt_solve_success,scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }


            // compute final step length and update the current iterate
            // --------------
            α = self.get_step_length(StepDirection::Combined,scaling);

            // check for undersized step and update strategy
            match self.strategy_checkpoint_small_step(α, scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }

            // Copy previous iterate in case the next one is a dud
            self.info.save_prev_iterate(&self.variables,&mut self.prev_vars);

            self.variables.add_step(&self.step_lhs, α);

        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

    // Check we if actually took a final step.  If not, we need
    // to recapture the scalars and print one last line
    if α == T::zero() {
        self.info.save_scalars(μ, α, σ, iter);
        notimeit! {timers; {self.info.print_status(&self.settings).unwrap();}}
    }

    timeit! {timers => "post-process"; {
            //check for "almost" convergence case and then extract solution
            self.info.post_process(&self.residuals, &self.settings);
            self.solution
                .post_process(&self.data, &mut self.variables, &self.info, &self.settings);
        }}

    //halt timers
    self.info.finalize(&mut timers);
    self.solution.finalize(&self.info);

    self.info.print_footer(&self.settings).unwrap();

    //stow the timers back into Option in the solver struct
    self.timers.replace(timers);
}
```
The `solve` function uses all the above trait implementations at some point in the loop, 
So in order to start re-implementation I would suggest starting with using the current default 
trait implementations found in `src/solver/implementations/default` and map each implementation to the
language you want to reimplement in(how you structure your new implementation will depend on the language which
I don't know). 

# for example the `Risiduals` trait   

```rust

/// Residuals for a conic optimization problem.

pub trait Residuals<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;

    /// Compute residuals for the given variables.
    ///
    fn update(&mut self, variables: &Self::V, data: &Self::D);
}
```

It's default Implementation
```rust

// ---------------
// Residuals type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Residuals`](crate::solver::core::traits::Residuals) trait

pub struct DefaultResiduals<T> {
    // the main KKT residuals
    pub rx: Vec<T>,
    pub rz: Vec<T>,
    pub rτ: T,

    // partial residuals for infeasibility checks
    pub rx_inf: Vec<T>,
    pub rz_inf: Vec<T>,

    // various inner products.
    // NB: these are invariant w.r.t equilibration
    pub dot_qx: T,
    pub dot_bz: T,
    pub dot_sz: T,
    pub dot_xPx: T,

    // the product Px by itself. Required for infeasibilty checks
    pub Px: Vec<T>,
}

impl<T> DefaultResiduals<T>
where
    T: FloatT,
{
    pub fn new(n: usize, m: usize) -> Self {
        let rx = vec![T::zero(); n];
        let rz = vec![T::zero(); m];
        let rτ = T::one();

        let rx_inf = vec![T::zero(); n];
        let rz_inf = vec![T::zero(); m];

        let Px = vec![T::zero(); n];

        Self {
            rx,
            rz,
            rτ,
            rx_inf,
            rz_inf,
            Px,
            dot_qx: T::zero(),
            dot_bz: T::zero(),
            dot_sz: T::zero(),
            dot_xPx: T::zero(),
        }
    }
}

impl<T> Residuals<T> for DefaultResiduals<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;

    fn update(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        // various products used multiple times
        let qx = data.q.dot(&variables.x);
        let bz = data.b.dot(&variables.z);
        let sz = variables.s.dot(&variables.z);

        //Px = P*x, P treated as symmetric
        let symP = data.P.sym();
        symP.symv(&mut self.Px, &variables.x, T::one(), T::zero());

        let xPx = variables.x.dot(&self.Px);

        //partial residual calc so we can check primal/dual
        //infeasibility conditions

        //Same as:
        //rx_inf .= -data.A'* variables.z
        let At = data.A.t();
        At.gemv(&mut self.rx_inf, &variables.z, -T::one(), T::zero());

        //Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
        self.rz_inf.copy_from(&variables.s);
        let A = &data.A;
        A.gemv(&mut self.rz_inf, &variables.x, T::one(), T::one());

        //complete the residuals
        //rx = rx_inf - Px - qτ
        self.rx.waxpby(-T::one(), &self.Px, -variables.τ, &data.q);
        self.rx.axpby(T::one(), &self.rx_inf, T::one());

        // rz = rz_inf - bτ
        self.rz
            .waxpby(T::one(), &self.rz_inf, -variables.τ, &data.b);

        // τ = qz + bz + κ + xPx/τ;
        self.rτ = qx + bz + variables.κ + xPx / variables.τ;

        //save local versions
        self.dot_qx = qx;
        self.dot_bz = bz;
        self.dot_sz = sz;
        self.dot_xPx = xPx;
    }
}

```

