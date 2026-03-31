:github_url: https://github.com/kevinzakka/mink/tree/main/docs/tutorial/example_gallery.rst

.. _example-gallery:

===============
Example Gallery
===============

The following examples demonstrate inverse-kinematics tasks of increasing complexity, from single-arm pose tracking to whole-body humanoid control.

Franka Panda
------------

End-effector pose tracking with posture regularization on a 7-DOF manipulator.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/arm_panda.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/arm_panda.py>`__

UR5e with Collision Avoidance
-----------------------------

Pose tracking with collision avoidance between the wrist and environment
(floor, walls). Demonstrates :class:`~mink.CollisionAvoidanceLimit`.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/arm_ur5e.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/arm_ur5e.py>`__

Bimanual manipulation with ALOHA
--------------------------------

Coordinated bimanual pose tracking with the ALOHA robot.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/arm_aloha.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/arm_aloha.py>`__

Dexterous hand + arm
--------------------

Combined arm and dexterous-hand control with an Allegro hand mounted on a KUKA IIWA arm.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/arm_hand_iiwa_allegro.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/arm_hand_iiwa_allegro.py>`__

Mobile Manipulation with TidyBot
--------------------------------

Mobile-base pose tracking with Stanford's TidyBot platform.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/mobile_tidybot.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/mobile_tidybot.py>`__

Humanoid
--------

Whole-body control with a Unitree G1 humanoid.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/humanoid_g1.mp4" type="video/mp4">
   </video>

`View source code <https://github.com/kevinzakka/mink/blob/main/examples/humanoid_g1.py>`__
