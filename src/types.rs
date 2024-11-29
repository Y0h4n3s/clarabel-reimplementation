#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum ScalingStrategy {
    PrimalDual,
    Dual,
}
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum StrategyCheckpoint {
    Update(ScalingStrategy), // Checkpoint is suggesting a new ScalingStrategy
    NoUpdate,                // Checkpoint recommends no change to ScalingStrategy
    Fail,                    // Checkpoint found a problem but no more ScalingStrategies to try
}

