# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Added `utils.get_subtree_body_ids` to get all bodies belonging to the subtree starting at a given body.
  - Example usage of this function can be found in the [ALOHA example script](examples/arm_aloha.py) where it is used to selectively apply gravity compensation torques to the left and right arm bodies.

### Changed

- Improved ALOHA example script.

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
