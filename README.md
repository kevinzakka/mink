# mink

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mink/ci.yml?branch=main)](https://github.com/kevinzakka/mink/actions)
[![Coverage Status](https://coveralls.io/repos/github/kevinzakka/mink/badge.svg)](https://coveralls.io/github/kevinzakka/mink?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/mink)](https://pypi.org/project/mink/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mink?color=blue)](https://pypistats.org/packages/mink)
![Banner for mink](https://github.com/kevinzakka/mink/blob/assets/banner.png?raw=true)

mink is a library for differential inverse kinematics in Python, based on the [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine.

Features include:

* Task specification in configuration or operational space;
* Limits on joint positions and velocities;
* Collision avoidance between any geom pair;
* Support for closed-chain kinematics (loop closures) via [equality constraints](https://mujoco.readthedocs.io/en/stable/computation/index.html#coequality);
* Lie group interface for rigid body transformations, with native C optimization for IK hot paths.

For usage and API reference, see the [documentation](https://kevinzakka.github.io/mink/).

If you use mink in your research, please cite it as follows:

```bibtex
@software{Zakka_Mink_Python_inverse_2026,
  author = {Zakka, Kevin},
  title = {{Mink: Python inverse kinematics based on MuJoCo}},
  year = {2026},
  month = feb,
  version = {1.1.0},
  url = {https://github.com/kevinzakka/mink},
  license = {Apache-2.0}
}
```

## Installation

Install from PyPI:

```bash
uv add mink
```

Or clone and run locally:

```bash
git clone https://github.com/kevinzakka/mink.git && cd mink
uv sync
```

Development dependencies require Python 3.10.12 or later. Tests and examples
use the pinned `mujoco-menagerie` package for robot assets. Local example XML
files preserve custom scenes, actuators, and robot combinations.

The first use of a packaged model downloads its assets and requires network
access. Later runs reuse the per-user cache and can run without network access.
Set `MENAGERIE_CACHE_DIR` to select a different cache directory.

## Examples

To run an example:

```bash
# Linux
uv run examples/arm_ur5e.py

# macOS
./fix_mjpython_macos.sh  # So that mjpython works with uv.
uv run mjpython examples/arm_ur5e.py
```

mink works with a variety of robots, including:

* **Single arms**: [Franka Panda](https://github.com/kevinzakka/mink/blob/main/examples/arm_panda.py), [UR5e](https://github.com/kevinzakka/mink/blob/main/examples/arm_ur5e.py), [KUKA iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/arm_iiwa.py), [ALOHA 2](https://github.com/kevinzakka/mink/blob/main/examples/arm_aloha.py)
* **Dual arms**: [Dual Panda](https://github.com/kevinzakka/mink/blob/main/examples/dual_panda.py), [Dual iiwa14](https://github.com/kevinzakka/mink/blob/main/examples/dual_iiwa.py), [Flying Dual UR5e](https://github.com/kevinzakka/mink/blob/main/examples/flying_dual_arm_ur5e.py)
* **Arm + hand**: [iiwa14 + Allegro](https://github.com/kevinzakka/mink/blob/main/examples/arm_hand_iiwa_allegro.py), [xArm + LEAP](https://github.com/kevinzakka/mink/blob/main/examples/arm_hand_xarm_leap.py)
* **Dexterous hands**: [Shadow Hand](https://github.com/kevinzakka/mink/blob/main/examples/hand_shadow.py)
* **Humanoids**: [Unitree G1](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_g1.py), [Unitree H1](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_h1.py), [Apptronik Apollo](https://github.com/kevinzakka/mink/blob/main/examples/humanoid_apollo.py)
* **Legged robots**: [Unitree Go1](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_go1.py), [Boston Dynamics Spot](https://github.com/kevinzakka/mink/blob/main/examples/quadruped_spot.py), [Agility Cassie](https://github.com/kevinzakka/mink/blob/main/examples/biped_cassie.py)
* **Mobile manipulators**: [TidyBot](https://github.com/kevinzakka/mink/blob/main/examples/mobile_tidybot.py), [Hello Robot Stretch](https://github.com/kevinzakka/mink/blob/main/examples/mobile_stretch.py), [Kinova Gen3 + LEAP](https://github.com/kevinzakka/mink/blob/main/examples/mobile_kinova_leap.py)

Check out the [examples](https://github.com/kevinzakka/mink/blob/main/examples/) directory for more.

## How can I help?

Install the library, use it and report any bugs in the [issue tracker](https://github.com/kevinzakka/mink/issues) if you find any. If you're feeling adventurous, you can also check out the contributing [guidelines](CONTRIBUTING.md) and submit a pull request.

## Acknowledgements

mink is a direct port of [Pink](https://github.com/stephane-caron/pink) which uses [Pinocchio](https://github.com/stack-of-tasks/pinocchio) under the hood. Stéphane Caron, the author of Pink, is a role model for open-source software in robotics. This library would not have been possible without his work and assistance throughout this project.

mink also heavily adapts code from the following libraries:

* The lie algebra library that powers the transforms in mink is adapted from [jaxlie](https://github.com/brentyi/jaxlie).
* The collision avoidance constraint is adapted from [dm_robotics](https://github.com/google-deepmind/dm_robotics/tree/main/cpp/controllers)'s LSQP controller.

## References

- The code implementing operations on matrix Lie groups contains references to numbered equations from the following paper: _[A micro lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537), Joan Solà, Jeremie Deray, and Dinesh Atchuthan, arXiv preprint arXiv:1812.01537 (2018)_.
