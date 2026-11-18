"""Tests for support_polygon_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest

from mink import ComTask, Configuration, solve_ik
from mink.exceptions import InvalidFrame, LimitDefinitionError
from mink.limits import SupportPolygonLimit

# A square support polygon, 0.4 m wide, centered on the frame's origin.
_SQUARE = np.array([[0.2, 0.2], [-0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]])

# A planar mobile base (x, y, yaw) carrying a two-link arm that pitches. With the
# arm upright, the CoM lies above the base's origin.
_MOBILE_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".2 .2 .05" mass="10"/>
      <site name="support"/>
      <body name="link1" pos="0 0 .05">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".03" mass="2"/>
        <body name="link2" pos="0 0 .5">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 .5" size=".03" mass="1"/>
        </body>
      </body>
    </body>
    <body name="empty" pos="5 0 0"/>
  </worldbody>
</mujoco>
"""

# A floating torso with a swinging arm; its polygon is given in the world frame.
_FLOATING_XML = """
<mujoco>
  <worldbody>
    <body name="torso" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .2" mass="5"/>
      <geom name="foot" type="box" size=".05 .05 .01" pos=".05 0 -.2"
            euler="10 20 30" mass=".1"/>
      <body name="arm" pos="0 .1 .1">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .4 0 0" size=".03" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A fixed body and a cart on a rail: the CoM's x is q / 2, linear in q.
_CART_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <geom type="sphere" size=".05" mass="1"/>
      <body name="cart">
        <joint name="rail" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _random_tangent(model: mujoco.MjModel, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, model.nv)


class TestSupportPolygonLimit(absltest.TestCase):
    """Test the support polygon limit."""

    @classmethod
    def setUpClass(cls):
        cls.mobile = mujoco.MjModel.from_xml_string(_MOBILE_XML)
        cls.floating = mujoco.MjModel.from_xml_string(_FLOATING_XML)
        cls.cart = mujoco.MjModel.from_xml_string(_CART_XML)

    def _mobile_limit(self, **kwargs) -> SupportPolygonLimit:
        return SupportPolygonLimit.from_vertices(
            self.mobile, "support", "site", _SQUARE, **kwargs
        )

    def test_invalid_definitions_raise(self):
        A, b = np.eye(2), np.ones(2)
        model = self.mobile
        bad = [
            lambda: SupportPolygonLimit(model, "support", "site", np.ones((2, 3)), b),
            lambda: SupportPolygonLimit(model, "support", "site", A, np.ones(3)),
            lambda: SupportPolygonLimit(
                model, "support", "site", np.zeros((0, 3)), np.zeros(0)
            ),
            lambda: SupportPolygonLimit(
                model, "support", "site", [[np.nan, 0.0], [0.0, 1.0]], b
            ),
            lambda: SupportPolygonLimit(model, "support", "site", A, [1.0, np.inf]),
            lambda: SupportPolygonLimit(
                model, "support", "site", [[0.0, 0.0], [0.0, 1.0]], b
            ),
            lambda: SupportPolygonLimit(model, "support", "site", A, b, margin=np.inf),
            lambda: SupportPolygonLimit(model, "support", "site", A, b, gain=0.0),
            lambda: SupportPolygonLimit(model, "support", "site", A, b, gain=1.5),
            lambda: SupportPolygonLimit(
                model, "support", "site", A, b, subtree_body="missing"
            ),
            lambda: SupportPolygonLimit(
                model, "support", "site", A, b, subtree_body=99
            ),
            lambda: SupportPolygonLimit(
                model, "support", "site", A, b, subtree_body="empty"
            ),
        ]
        for i, make in enumerate(bad):
            with self.assertRaises(LimitDefinitionError, msg=f"case {i}"):
                make()
        with self.assertRaises(InvalidFrame):
            SupportPolygonLimit(self.mobile, "missing", "site", A, b)

    def test_from_vertices_matches_polygon(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.3, 0.3, (200, 2))
        inside = np.all(np.abs(points) <= 0.2, axis=1)
        # Clockwise, counter-clockwise, and with an interior and repeated vertex.
        for vertices in (_SQUARE, _SQUARE[::-1], np.vstack([_SQUARE, [0, 0]] * 2)):
            limit = SupportPolygonLimit.from_vertices(
                self.mobile, "support", "site", vertices
            )
            self.assertEqual(limit.A.shape, (4, 2))
            np.testing.assert_allclose(np.linalg.norm(limit.A, axis=1), 1.0)
            np.testing.assert_array_equal(
                np.all(points @ limit.A.T <= limit.b, axis=1), inside
            )

    def test_from_vertices_rejects_degenerate_polygons(self):
        for vertices in ([[0, 0], [1, 0]], [[0, 0], [1, 0], [2, 0]], [[0, 0, 0]]):
            with self.assertRaises(LimitDefinitionError):
                SupportPolygonLimit.from_vertices(
                    self.mobile, "support", "site", vertices
                )

    def test_empty_polygon_is_inactive(self):
        configuration = Configuration(self.mobile)
        for A in (np.zeros((0, 2)), []):
            limit = SupportPolygonLimit(self.mobile, "support", "site", A, [])
            constraint = limit.compute_qp_inequalities(configuration, 0.1)
            self.assertTrue(constraint.inactive)

    def test_rows_normalized(self):
        limit = SupportPolygonLimit(
            self.mobile, "support", "site", [[2.0, 0.0], [0.0, -3.0]], [0.4, 0.6]
        )
        np.testing.assert_allclose(limit.A, [[1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(limit.b, [0.2, 0.2])

    def test_distances_follow_the_base(self):
        """With the arm upright the CoM is above the base's origin, wherever the
        base is: every edge is 0.2 m away."""
        limit = self._mobile_limit(margin=0.05, gain=0.5)
        configuration = Configuration(self.mobile)
        for base in ([0.0, 0.0, 0.0], [1.0, -2.0, 0.7]):
            configuration.update(np.array([*base, 0.0, 0.0]))
            np.testing.assert_allclose(limit.compute_distances(configuration), 0.2)
            h = limit.compute_qp_inequalities(configuration, 0.1).h
            assert h is not None
            np.testing.assert_allclose(h, 0.5 * (0.2 - 0.05))

    def test_jacobian_matches_finite_differences(self):
        """G dq is the first-order decrease of the distances along dq, for site,
        body and geom frames, on a planar base and on a rotated floating base."""
        rng = np.random.default_rng(1)
        q_mobile = np.array([0.3, -0.1, 0.5, 0.4, -0.7])
        q_floating = np.zeros(self.floating.nq)
        q_floating[:3] = [0.2, -0.3, 1.0]
        q_floating[3:7] = rng.normal(size=4)
        q_floating[3:7] /= np.linalg.norm(q_floating[3:7])
        q_floating[7] = 0.4
        cases = [
            (self._mobile_limit(), q_mobile),
            (
                # A body frame on a pitched link.
                SupportPolygonLimit.from_vertices(
                    self.mobile, "link2", "body", _SQUARE
                ),
                q_mobile,
            ),
            (
                SupportPolygonLimit.from_vertices(
                    self.floating, "world", "body", _SQUARE, subtree_body="torso"
                ),
                q_floating,
            ),
            (
                # A rotated geom on the floating torso.
                SupportPolygonLimit.from_vertices(
                    self.floating, "foot", "geom", _SQUARE
                ),
                q_floating,
            ),
        ]
        eps = 1e-6
        for limit, q0 in cases:
            model = limit.model
            configuration = Configuration(model, q0)
            G = limit.compute_qp_inequalities(configuration, 0.1).G
            assert G is not None
            for _ in range(3):
                dq = _random_tangent(model, rng)
                d = []
                for sign in (1.0, -1.0):
                    q = q0.copy()
                    mujoco.mj_integratePos(model, q, sign * dq, eps)
                    configuration.update(q)
                    d.append(limit.compute_distances(configuration))
                np.testing.assert_allclose(
                    (d[0] - d[1]) / (2 * eps), -G @ dq, atol=1e-8
                )
            configuration.update(q0)

    def test_planar_base_does_not_enter(self):
        limit = self._mobile_limit()
        configuration = Configuration(self.mobile)
        configuration.update(np.array([0.3, -0.1, 0.5, 0.4, -0.7]))
        G = limit.compute_qp_inequalities(configuration, 0.1).G
        assert G is not None
        np.testing.assert_allclose(G[:, :3], 0.0, atol=1e-12)
        self.assertGreater(np.abs(G[:, 3:]).max(), 0.01)

    def test_clamp_violated(self):
        configuration = Configuration(self.mobile)
        configuration.update(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        # A margin of 0.3 m puts the CoM 0.1 m inside the band of every edge.
        retreat = self._mobile_limit(margin=0.3).compute_qp_inequalities(
            configuration, 0.1
        )
        hold = self._mobile_limit(
            margin=0.3, clamp_violated=True
        ).compute_qp_inequalities(configuration, 0.1)
        assert retreat.h is not None and hold.h is not None
        np.testing.assert_allclose(retreat.h, -0.1)
        np.testing.assert_array_equal(hold.h, 0.0)
        np.testing.assert_array_equal(retreat.G, hold.G)

    def _cart_step(self, q: float, target: float, **kwargs) -> float:
        """Take one IK step of a CoM task towards `target`; return the CoM's x."""
        configuration = Configuration(self.cart)
        configuration.update(np.array([q]))
        limit = SupportPolygonLimit.from_vertices(
            self.cart, "world", "body", _SQUARE, margin=0.05, **kwargs
        )
        task = ComTask(cost=1.0)
        task.set_target(np.array([target, 0.0, 0.0]))
        vel = solve_ik(configuration, [task], 1.0, "daqp", limits=[limit])
        configuration.integrate_inplace(vel, 1.0)
        return float(configuration.data.subtree_com[1][0])

    def test_solve_ik_stops_at_the_margin(self):
        """The CoM is linear in q here, so the limit holds exactly."""
        self.assertAlmostEqual(self._cart_step(0.0, 1.0), 0.15, places=6)

    def test_solve_ik_retreats_or_holds(self):
        """From a CoM 0.1 m beyond the margin (x = 0.25), the default limit moves
        it back to the margin; with clamp_violated it may stay."""
        self.assertAlmostEqual(self._cart_step(0.5, 0.25), 0.15, places=6)
        self.assertAlmostEqual(
            self._cart_step(0.5, 0.25, clamp_violated=True), 0.25, places=6
        )
        self.assertLessEqual(
            self._cart_step(0.5, 1.0, clamp_violated=True), 0.25 + 1e-9
        )


if __name__ == "__main__":
    absltest.main()
