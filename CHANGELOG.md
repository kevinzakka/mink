# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Add 3.8 and 3.13 to CI test matrix.
- Switch to `MjSpec` for model construction in examples and eliminate `dm_control` dependency.
- Added single and dual Franka Emika Panda robot examples featuring motion planning, and bi-manual coordination. Contribution from @Debojit-D.

![bimanual panda](https://github.com/kevinzakka/mink/blob/assets/dual_panda.gif?raw=true)

### Changed

- Merged task and limit exceptions to base exceptions file.
- Improved the documentation for certain tasks.

## [0.0.10] - 2025-04-22

### Changed

- Update so3.py log function to use arctan2 instead of arccos for better numerical stability.

## [0.0.9] - 2025-04-21

### Changed

- Switch Lie algebra implementation to use mujoco functions.
- Rewrite some Lie algebra methods to use derivations with least operations.
- Relax tolerances on test_solve_ik.py test_single_task_convergence.
- Relax numpy version requirement.

## [0.0.8] - 2025-04-21

### Added

- Added equality operators for SE3 and SO3.
- Added matrix Lie group interpolation.

### Changed

- Remove quadprog dependency and switch to `daqp` for examples.

## [0.0.7] - 2025-03-28

### Added

- Added support for Python 3.8.

## [0.0.6] - 2025-03-15

### Added

- Added `EqualityConstraintTask`, which is particularly useful for closed-chain mechanisms like 4-bar linkages or parallel robots. See [biped_cassie.py](examples/biped_cassie.py) for an implementation example. Contribution from @simeon-ned.

| Before | After |
|--------|-------|
| ![before](https://github.com/kevinzakka/mink/blob/assets/equality_before.gif?raw=true) | ![after](https://github.com/kevinzakka/mink/blob/assets/equality_after.gif?raw=true) |

- `Configuration.check_limits` now logs joint limit violations to `debug` rather than `warn` when `safety_break=False`.
- Added `utils.get_subtree_body_ids` to get all bodies belonging to the subtree starting at a given body.
  - Example usage of this function can be found in the [ALOHA example script](examples/arm_aloha.py) where it is used to selectively apply gravity compensation torques to the left and right arm bodies.
- Add G1 and Apollo humanoid example with a tabletop manipulation focus.

![g1 teleop](https://github.com/kevinzakka/mink/blob/assets/g1_teleop.gif?raw=true)

### Changed

- Improved ALOHA example script.
- Fixed a small bug in the exit criterion of the ALOHA example script.
- Updated Stretch example to work with MuJoCo >= 3.3.0.

## [0.0.5] - 2024-09-27

### Changed

- Changed `inf` to `mujoco.mjMAXVAL` for unbounded joint limits in `ConfigurationLimit`.

## [0.0.4] - 2024-09-26

### Changed

- Fixed a bug in `ConfigurationLimit` where the indices of the limited DoFs were storing the wrong values.
- Changed `print` statement in `Configuration::check_limits` to `logging.warning`.

### Added

- Examples:
    - Mobile Kinova loaded from a module-oriented MJCF description: [mobile_kinova.py](examples/mobile_kinova.py) (thanks @Zi-ang-Cao)
        - Removes the nested class naming convention in [tidybot.xml](tidybot.xml), allowing for seamless swapping of end-effectors.
    - Mobile Kinova with LEAP hand: [mobile_kinova_leap.py](examples/mobile_kinova_leap.py) (thanks @Zi-ang-Cao)
        - The example scripts also includes updated recommendations for controlling the mobile base, aiming to minimize unnecessary rotation.
    - UFactory xArm7 with LEAP hand: [xarm_leap.py](examples/arm_hand_xarm_leap.py)
    - Unitree H1: [humanoid_h1.py](examples/humanoid_h1.py)

## [0.0.3] - 2024-08-10

### Added

- Relative frame task.
- More examples:
    - Mobile manipulator: [mobile_tidybot.py](examples/mobile_tidybot.py)
    - Bimanual manipulator: [aloha.py](examples/arm_aloha.py)
    - Arm + dexterous hand: [arm_hand_iiwa_allegro.py](examples/arm_hand_iiwa_allegro.py)

### Changed

- Posture task cost can now be a scalar or a vector. Vector costs are useful for specifying different costs for different dofs.

## [0.0.2] - 2024-07-27

### Added

- [Documentation](https://kevinzakka.github.io/mink/).
- Damping task.

### Changed

- Restrict numpy version to < 2.0.0 for compatibility with `osqp` solver in `qpsolvers`.
- README touchup.

## [0.0.1] - 2024-07-25

Initial release.
