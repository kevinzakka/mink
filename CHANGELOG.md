# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Examples:
    - UFactory xArm7 with LEAP hand: [xarm_leap.py](examples/arm_hand_xarm_leap.py)

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
