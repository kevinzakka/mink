# mink

A port of [pink](https://github.com/stephane-caron/pink) for MuJoCo.

## Installation

```bash
pip install -e .
```

## Examples

```python
mjpython examples/arm_ur5e.py  # On macOS
python examples/arm_ur5e.py  # On Linux
```

## Background

Yes, you can use your box-constrained QP solver to solve a problem with additional linear constraints of the form \( Ax \leq b \) by introducing slack variables. Slack variables can transform the general linear inequality constraints into box constraints that are compatible with your solver.

Here’s a step-by-step approach to incorporating slack variables:

### Step-by-Step Process

1. **Introduce Slack Variables**:
   For each inequality constraint \( a_i^T x \leq b_i \), introduce a non-negative slack variable \( s_i \) such that:
   \[ a_i^T x + s_i = b_i \]
   Here, \( s_i \geq 0 \). This transforms the inequality constraint into an equality constraint with an additional variable.

2. **Reformulate the Problem**:
   Extend the original decision vector \( x \) to include the slack variables \( s \). Let's denote the new decision vector as \( z = [x; s] \). The problem now becomes:
   \[ \min \frac{1}{2} x^T Q x + c^T x \]
   subject to
   \[ l \leq x \leq u \]
   \[ s \geq 0 \]
   \[ A x + s = b \]

   Here, the equality \( A x + s = b \) can be split into two inequalities \( A x + s \leq b \) and \( A x + s \geq b \).

3. **Incorporate Slack Variables into Box Constraints**:
   To fit into a box-constrained QP format, recognize that the slack variables are subject to non-negativity constraints \( s \geq 0 \). This can be rewritten as:
   \[ 0 \leq s \]

   Since \( s \) are new variables added to the optimization, their upper bounds are not explicitly needed unless specified by your problem context.

4. **Extend the Quadratic and Linear Terms**:
   The original quadratic objective \( \frac{1}{2} x^T Q x + c^T x \) doesn't directly include the slack variables. If we assume the slack variables do not contribute to the quadratic part of the objective, then the extended problem with \( z = [x; s] \) remains:
   \[ \min \frac{1}{2} x^T Q x + c^T x \]

   You can introduce a zero matrix for the slack variables to maintain the structure:
   \[ \frac{1}{2} z^T \begin{bmatrix} Q & 0 \\ 0 & 0 \end{bmatrix} z + \begin{bmatrix} c \\ 0 \end{bmatrix}^T z \]

   Here, \( Q \) and \( c \) are expanded to include zeros corresponding to the slack variables, ensuring they do not influence the objective function unless specified.

5. **Solve the Extended Problem**:
   You now have a box-constrained QP problem with the extended decision vector \( z \), which includes both the original variables \( x \) and the slack variables \( s \), subject to the constraints:
   \[ l_x \leq x \leq u_x \]
   \[ 0 \leq s \]
   \[ A x + s = b \]

   This fits within the framework of a box-constrained QP solver.

### Summary
To solve your original problem with additional linear constraints using a box-constrained QP solver, you can:

1. Introduce slack variables \( s \) to convert the inequalities into equalities.
2. Extend the decision vector and constraints to include these slack variables.
3. Reformulate the objective function and constraints to fit into the box-constrained QP framework.

This way, you leverage your existing solver to handle the additional constraints without changing the nature of the problem significantly.


## References

https://hal.science/hal-04621130/file/OpenSoT_journal_wip.pdf
