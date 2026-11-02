# Unitree H2 with Sharpa Wave hands (MJCF)

## Overview

This package contains a kinematic description (MJCF) of the [Unitree H2](https://www.unitree.com/h2) humanoid with two [Sharpa Wave](https://github.com/google-deepmind/mujoco_menagerie/tree/main/sharpa_wave) hands in place of its own.

* The H2 body, joints, limits and inertias come from the H2 description in [unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/h2_description), licensed under BSD-3-Clause (see `LICENSE-unitree`).
* The Sharpa Wave hands come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/sharpa_wave), licensed under Apache-2.0 (see `LICENSE-sharpa`).

## Modifications

1. Converted the H2 URDF to MJCF and added a free joint at the pelvis, with the soles on the floor.
2. Removed the H2 hands and mounted a Sharpa Wave hand at each wrist flange.
3. Split the Sharpa hand meshes by material, and added sole boxes and sites to each foot.
4. Removed the actuators: the model is for kinematics.
