# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- Bumped minimum `mujoco` to 3.10.0 and switched `Configuration.get_inertia_matrix` to the new `mj_fullM(model, data, dst)` signature, replacing the `mju_sym2dense` workaround (#149). Type stubs regenerated against 3.11.0.

### Fixed

- `EqualityConstraintTask`: costs are now looked up by position in the selected subset instead of by equality id. Previously `equalities=[0, 3]` raised `IndexError` on the first solve and out-of-order subsets mapped costs to the wrong constraints.
- `ConfigurationLimit` / `Configuration.check_limits`: a limited ball joint is now constrained on its total rotation angle (`range[1]`), matching MuJoCo semantics. Previously its quaternion qpos was box-constrained against the range and `check_limits` compared the quaternion w component, so violations went undetected. See `examples/googly_eyes.py` for a demo.

| With `ConfigurationLimit` | Without |
| :---: | :---: |
| ![](https://github.com/kevinzakka/mink/blob/assets/googly_eyes.gif?raw=true) | ![](https://github.com/kevinzakka/mink/blob/assets/googly_eyes_no_limit.gif?raw=true) |

### Added

- `FreeJointVelocityLimit`: bounds the linear and angular velocity of a free joint (floating base), which `VelocityLimit` ignores.

- `LookAtTask`: points a frame's local axis at a target position in the world. It is a two-DOF task that constrains only the line of sight and leaves roll about the gaze axis free, so unlike a `FrameTask` orientation target it does not over-constrain the orientation. Its Jacobian is derived in the local frame from the body Jacobian and includes the translation term, so the solver may translate the frame as well as rotate it to keep the target in view.

| UR5e wrist camera | Apollo whole-body gaze |
| :---: | :---: |
| ![](https://github.com/kevinzakka/mink/blob/assets/arm_ur5e_wrist_cam_lookat.gif?raw=true) | ![](https://github.com/kevinzakka/mink/blob/assets/humanoid_apollo.gif?raw=true) |

- `AxisAlignTask`: aligns a frame's local axis with a target *direction* in the world (a surface normal, gravity, a machine axis). It is the sibling of `LookAtTask`: look-at points an axis at a point, axis-align points it along a direction at infinity. Like look-at it is a two-DOF task that leaves roll about the axis free, but its Jacobian has no translation term since only rotation changes a direction. See `examples/arm_panda_engrave.py`, where a Panda holds the spindle normal to a curved surface while engraving.

| Panda engraving the MuJoCo "M" |
| :---: |
| ![](https://github.com/kevinzakka/mink/blob/assets/arm_panda_engrave.gif?raw=true) |

### Removed

- `examples/humanoid_h1.py` and the `unitree_h1` model assets, whose whole-body tracking is already covered by `examples/humanoid_g1.py`.

## [1.2.0] - 2026-06-20

### Changed

- `CollisionAvoidanceLimit`: added a bounding-sphere/plane broadphase that skips geom pairs out of `collision_detection_distance` range before the narrow-phase `mj_geomDistance` query, mirroring MuJoCo's `mj_filterSphere`. The assembled constraint is unchanged. ~2.9x faster collision phase, ~3.3x faster end-to-end IK on the ALOHA dual-arm scene (M1 Max). Disable with `broadphase=False`.
- `solve_ik` / `build_ik`: fused the QP objective assembly. Tasks now expose a weighted least-squares residual (`Task.compute_qp_residual`) that the solver stacks into a single `WᵀW` matmul instead of summing per-task Hessians, and `Configuration` caches resolved frame ids. ~1.3x faster end-to-end IK on the G1 humanoid (8 tasks), scaling with task count (~1.4x on a 21-task hand, M1 Max). The assembled objective is unchanged up to floating-point summation order.

## [1.1.1] - 2026-05-15

### Added

- `get_subtree_joint_ids` utility for retrieving all joints in the subtree rooted at a given body. Contribution from @maxstrobel.
- `get_body_joint_ids` utility for retrieving immediate joints belonging to a given body. Contribution from @maxstrobel.

### Changed

- `get_subtree_geom_ids` and `get_subtree_joint_ids` now reuse `get_subtree_body_ids` instead of duplicating the traversal logic. Contribution from @maxstrobel.
- Bumped minimum `mujoco` to `3.8.1` and migrated `Configuration.get_inertia_matrix` off the soon-to-break `mj_fullM` to `mju_sym2dense` on the CSR-format `data.M`.

### Fixed

- `compute_look_at_rotation` in `examples/humanoid_apollo.py` no longer divides by zero when the look target coincides with the source, which previously produced NaNs in the IK solver (#135).

## [1.1.0] - 2026-02-19

No public API changes — this is a drop-in upgrade.

### Added

- Native C extension (`_lie_ops_c`) for SE3 operations in the IK hot path, with automatic fallback to pure Python.
  - 8 fused operations (`se3_log`, `se3_rminus`, `se3_inverse_multiply`, `se3_jlog`, `se3_adjoint`, etc.) on raw double arrays, no intermediate Python objects.
  - `FrameTask`, `RelativeFrameTask`, and `Configuration` pass raw `ndarray[7]` internally for end-to-end native path.
  - Disable with `MINK_DISABLE_NATIVE=1`.
- Build backend switched to `scikit-build-core`. C extension is optional for source installs without a compiler.

### Changed

- `CollisionAvoidanceLimit`: pre-allocated scratch buffers, inlined hot loop, removed per-pair `Contact` dataclass. ~12x faster.
- `Configuration.check_limits`: vectorized using precomputed joint indices and numpy ops.

## 1.0.0 - 2025-12-19

### Changed

- Switch to `uv`.
- Deprecate Python 3.8 and 3.9.

### Added

- Associate loggers with the name "mink".
- Improve test coverage.
- Add more tests to `test_configuration_limit.py` and `test_velocity_limit.py`.
- Add support for equality constraints in `solve_ik` and `build_ik` via the `constraints` parameter.
- Add `DofFreezingTask` to freeze specific degrees of freedom using equality constraints.

## [0.0.13] - 2025-09-12

### Bugfix

* Objects already in collision repel each other instead of locking together. Contribution from @mattrobotcontributor.

## [0.0.12] - 2025-08-08

### Changed

- `solve_ik` now raises a new exception, `NoSolutionFound`, when the QP solver fails to find a solution.

### Bugfix

* Fix `Configuration.get_inertia_matrix` for MuJoCo versions >= 3.3.4.

### Added

- `clamp` method for SO3 and SE3. Contribution from @adlarkin.

## [0.0.11] - 2025-05-22

### Added

- Added `KineticEnergyRegularizationTask`, which regularizes joint displacements based on kinetic energy (inertia-weighted damping).
  - New example: [examples/kinetic_energy_reg.py](examples/kinetic_energy_reg.py).
- Add `Configuration.get_inertia_matrix()`: returns the joint-space inertia matrix at the current configuration.
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
