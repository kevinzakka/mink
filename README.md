# mink

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mink/ci.yml?branch=main)](https://github.com/kevinzakka/mink/actions)
[![Coverage Status](https://coveralls.io/repos/github/kevinzakka/mink/badge.svg)](https://coveralls.io/github/kevinzakka/mink?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/mink)](https://pypi.org/project/mink/)
![Banner for mink](https://github.com/kevinzakka/mink/blob/assets/banner.png?raw=true)

mink is a library for differential inverse kinematics in Python, based on the [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine.

Features include:

* Task specification in configuration or operational space;
* Limits on joint positions and velocities;
* Collision avoidance between any geom pair;
* Lie group interface for rigid body transformations.

For usage and API reference, see the [documentation](https://kevinzakka.github.io/mink/).

If you use mink in your research, please cite it as follows:

```bibtex
@software{Zakka_Mink_Python_inverse_2024,
  author = {Zakka, Kevin},
  license = {Apache-2.0},
  month = jul,
  title = {{Mink: Python inverse kinematics based on MuJoCo}},
  url = {https://github.com/kevinzakka/mink},
  version = {0.0.4},
  year = {2024}
}
```

## Installation

You can install `mink` using `pip`:

```bash
pip install mink
```

To include the example dependencies:

```bash
pip install "mink[examples]"
```

## Examples

mink works with a variety of robots, including:

* Arms: [UR5e](https://github.com/kevinzakka/mink/blob/main/examples/arm_ur5e_actuators.py), [iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/arm_iiwa.py), [bimanual iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/dual_iiwa.py)
* Humanoids: [Unitree G1](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_g1.py), [Unitree H1](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_h1.py)
* Quadrupeds: [Unitree Go1](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_go1.py), [Boston Dynamics Spot](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_spot.py)
* Hands: [Shadow Hand](https://github.com/kevinzakka/mink/blob/main/examples/hand_shadow.py), [Allegro Hand](https://github.com/kevinzakka/mink/blob/main/examples/arm_hand_iiwa_allegro.py)
* Mobile manipulators: [Stanford TidyBot](https://github.com/kevinzakka/mink/blob/main/examples/mobile_tidybot.py), [Hello Robot Stretch](https://github.com/kevinzakka/mink/blob/main/examples/mobile_stretch.py)

Check out the [examples](https://github.com/kevinzakka/mink/blob/main/examples/) directory for more code.

## How can I help?

Install the library, use it and report any bugs in the [issue tracker](https://github.com/kevinzakka/mink/issues) if you find any. If you're feeling adventurous, you can also check out the contributing [guidelines](CONTRIBUTING.md) and submit a pull request.

## Acknowledgements

mink is a direct port of [Pink](https://github.com/stephane-caron/pink) which uses [Pinocchio](https://github.com/stack-of-tasks/pinocchio) under the hood. Stéphane Caron, the author of Pink, is a role model for open-source software in robotics. This library would not have been possible without his work and assistance throughout this project.

mink also heavily adapts code from the following libraries:

* The lie algebra library that powers the transforms in mink is adapted from [jaxlie](https://github.com/brentyi/jaxlie).
* The collision avoidance constraint is adapted from [dm_robotics](https://github.com/google-deepmind/dm_robotics/tree/main/cpp/controllers)'s LSQP controller.
