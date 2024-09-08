# Tidybot and Mobile Kinova Description (MJCF)

## Overview

This package introduces a mobile manipulator developed by the [Interactive Perception and Robot Learning Lab](https://iprl.stanford.edu). The robot features a Kinova Gen 3 arm mounted on a holonomic base, which has been used in the well-known [Tidybot](https://tidybot.cs.princeton.edu) project for room tidying tasks. To expand the robot's functionality, the default 2F85 Robotiq gripper can be replaced with more dexterous hands, such as the [Leap Hand](https://leaphand.com/). The package provides a modular MJCF description of the mobile Kinova robot, enabling seamless swapping of end-effectors. Please note the nested class naming convention in the [tidybot.xml](tidybot.xml) file. For compatibility, this convention has been removed in the [mobile_kinova.xml](mobile_kinova.xml) file.

## Examples

In the example script [mobile_tidybot.py](../mobile_tidybot.py), the robot equipped with the 2F85 Robotiq gripper is loaded from [tidybot.xml](tidybot.xml).
In the example scripts [mobile_kinova.py](../mobile_kinova.py) and [mobile_kinova_leap.py](../mobile_kinova_leap.py), the mobile Kinova robot is loaded from [mobile_kinova.xml](mobile_kinova.xml). Both scripts also include updated recommendations for controlling the mobile base, aiming to minimize unnecessary rotation.
