:github_url: https://github.com/kevinzakka/mink/tree/main/doc/derivation.rst

.. _derivations:

***********
Derivations
***********

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Symbol
   * - Configuration
     - :math:`q`
   * - Configuration displacement
     - :math:`\Delta q`
   * - Integration timestep
     - :math:`dt`
   * - Velocity in tangent space
     - :math:`v = \frac{\Delta q}{dt}`
   * - Configuration limits
     - :math:`q_{\text{min}}, q_{\text{max}}`
   * - Maximum joint velocity magnitude
     - :math:`v_{\text{max}}`
   * - Identity matrix
     - :math:`I`

------
Limits
------

Configuration limit
===================

Using a first-order Taylor expansion on the configuration, we can write the limit as:

.. math::
    \begin{aligned}
    q_{\text{min}} &\leq q \oplus v \cdot dt \leq q_{\text{max}} \\
    q_{\text{min}} &\leq q \oplus \Delta q \leq q_{\text{max}} \\
    q_{\text{min}} &\ominus q \leq \Delta q \leq q_{\text{max}} \ominus q
    \end{aligned}

Rewriting as :math:`G \Delta q \leq h`, we separate the inequalities:

.. math::
    \begin{aligned}
    &+I \cdot \Delta q \leq q_{\text{max}} \ominus q \\
    &-I \cdot \Delta q \leq q \ominus q_{\text{min}}
    \end{aligned}

Stacking these inequalities, we define:

.. math::
    \begin{aligned}
    G &= \begin{bmatrix} +I \\ -I \end{bmatrix}, \\
    h &= \begin{bmatrix} q_{\text{max}} \ominus q \\ q \ominus q_{\text{min}} \end{bmatrix}
    \end{aligned}

Velocity limit
==============

Given the maximum joint velocity magnitudes :math:`v_{\text{max}}`, the joint velocity limits can be expressed as:

.. math::
    \begin{aligned}
    -v_{\text{max}} &\leq v \leq v_{\text{max}} \\
    -v_{\text{max}} &\leq \frac{\Delta q}{dt} \leq v_{\text{max}} \\
    -v_{\text{max}} \cdot dt &\leq \Delta q \leq v_{\text{max}} \cdot dt
    \end{aligned}

Rewriting as :math:`G \Delta q \leq h`, we separate the inequalities:

.. math::
    \begin{aligned}
    &+I \cdot \Delta q \leq v_{\text{max}} \cdot dt \\
    &-I \cdot \Delta q \leq v_{\text{max}} \cdot dt
    \end{aligned}

Stacking these inequalities, we define:

.. math::
    \begin{aligned}
    G \Delta q &\leq h \\
    \begin{bmatrix} +I \\ -I \end{bmatrix} \Delta q &\leq \begin{bmatrix} v_{\text{max}} \cdot dt \\ v_{\text{max}} \cdot dt \end{bmatrix}
    \end{aligned}


-----
Tasks
-----
