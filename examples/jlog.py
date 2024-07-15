import time

import numpy as np
import pinocchio as pin
from manifpy import SE3

np.set_printoptions(precision=2, suppress=True, threshold=1e-5)

##
## Right minus
##

X = SE3.Random()
Y = SE3.Random()
X_pin = pin.SE3(X.transform())
Y_pin = pin.SE3(Y.transform())

# Pinnochio.
X_minus_Y_p = pin.log(Y_pin.actInv(X_pin))

# Manif.
X_minus_Y_m = X.minus(Y)

np.testing.assert_allclose(X_minus_Y_p.vector, X_minus_Y_m.coeffs())

pin_time = []
for _ in range(500):
    tic = time.time()
    X_minus_Y_pin = pin.log(Y_pin.actInv(X_pin))
    pin_time.append(time.time() - tic)
print(f"avg pin time: {np.mean(pin_time)}")

manif_time = []
for _ in range(500):
    tic = time.time()
    X_minus_Y = X.minus(Y)
    manif_time.append(time.time() - tic)
print(f"avg manif time: {np.mean(manif_time)}")
