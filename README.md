# mink

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mink/ci.yml?branch=main)](https://github.com/kevinzakka/mink/actions)

mink is a library for differential inverse kinematics in Python, based on the [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine.

## Installation

```bash
pip install -e ".[examples]"
```

## Examples

* Arms: [UR5e](https://github.com/kevinzakka/mink/blob/main/examples/arm_ur5e_actuators.py), [iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/arm_iiwa.py), [bimanual iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/dual_iiwa.py)
* Humanoids: [Unitree G1](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_g1.py)
* Quadrupeds: [Unitree Go1](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_go1.py), [Boston Dynamics Spot](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_spot.py)
* Hands: [Shadow Hand]()

Check out the [examples](https://github.com/kevinzakka/mink/blob/main/examples/) directory for more code.

## How can I help?

Install the library, use it and report any bugs in the [issue tracker](https://github.com/kevinzakka/mink/issues) if you find any. If you're feeling adventurous, you can also check out the contributing [guidelines](CONTRIBUTING.md) and submit a pull request.

## Acknowledgements

mink is a direct port of [pink](https://github.com/stephane-caron/pink) which uses [Pinocchio](https://github.com/stack-of-tasks/pinocchio) under the hood. Stéphane Caron, the original author of pink, is a role model for open-source software in robotics. This library would not have been possible without his work and assistance throughout this project.

mink also heavily adapts code from the following libraries:

* The lie algebra library that powers the transforms in mink is adapted from [jaxlie](https://github.com/brentyi/jaxlie).
* The collision avoidance constraint is adapted from [dm_robotics](https://github.com/google-deepmind/dm_robotics/tree/main/cpp/controllers)'s LSQP controller.
