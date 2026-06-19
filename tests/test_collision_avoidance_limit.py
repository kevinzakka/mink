"""Tests for collision_avoidance_limit.py."""

import itertools

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_geom_pairs_strings(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="ball" name="ball"/>
              <geom name="g1" type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57"/>
                <geom name="g2" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body>
              <joint type="hinge" name="hinge2" range="0 1.57"/>
              <geom name="g3" type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        limit = CollisionAvoidanceLimit(model=model, geom_pairs=[(["g1"], ["g3"])])
        self.assertListEqual(limit.geom_id_pairs, [(0, 2)])

    def test_geom_pairs_deduplication(self):
        """Test that duplicate geom pairs are deduplicated."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="ball" name="ball"/>
              <geom name="g1" type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57"/>
                <geom name="g2" type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
            <body>
              <joint type="hinge" name="hinge2" range="0 1.57"/>
              <geom name="g3" type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)

        # Case 1: Duplicate collision pairs.
        limit = CollisionAvoidanceLimit(
            model=model, geom_pairs=[(["g1"], ["g3"]), (["g1"], ["g3"])]
        )
        # Should have only one pair (0, 2), not two.
        self.assertListEqual(limit.geom_id_pairs, [(0, 2)])
        self.assertEqual(len(limit.geom_id_pairs), 1)

        # Case 2: Overlapping geom groups that produce duplicate pairs.
        limit = CollisionAvoidanceLimit(
            model=model, geom_pairs=[(["g1", "g2"], ["g3"]), (["g1"], ["g3"])]
        )
        # Should deduplicate the (0, 2) pair that appears from both collision pairs.
        self.assertEqual(len(limit.geom_id_pairs), 2)  # (0, 2) and (1, 2)
        self.assertIn((0, 2), limit.geom_id_pairs)
        self.assertIn((1, 2), limit.geom_id_pairs)

        # Case 3: Reversed order should also be deduplicated.
        limit = CollisionAvoidanceLimit(
            model=model, geom_pairs=[(["g1"], ["g3"]), (["g3"], ["g1"])]
        )
        # Both should produce (0, 2) due to min/max normalization.
        self.assertListEqual(limit.geom_id_pairs, [(0, 2)])
        self.assertEqual(len(limit.geom_id_pairs), 1)

    def test_dimensions(self):
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Check that non-colliding geoms are correctly filtered out and that we have
        # the right number of max expected contacts.
        g1_coll = [
            g
            for g in g1
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        g2_coll = [
            g
            for g in g2
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)
        assert G is not None and h is not None

        # The upper bound should always be >= relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check that the inequality constraint dimensions are valid.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        model = load_robot_description("ur5e_mj_description")
        nv = model.nv

        # Options necessary to obtain separation normal + dense matrices.
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

        # Remove unnecessary constraints.
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT

        # Set contact dimensionality to 1 (normals only).
        model.geom_condim[:] = 1

        data = mujoco.MjData(model)

        # Handcrafted qpos with multiple contacts.
        qpos_coll = np.asarray([-1.5708, -1.5708, 3.01632, -1.5708, -1.5708, 0])
        data.qpos = qpos_coll
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 1)

        for i in range(data.ncon):
            # Get MuJoCo's contact normal jacobian.
            contact = data.contact[i]
            start_idx = contact.efc_address * nv
            end_idx = start_idx + nv
            efc_J = data.efc_J[start_idx:end_idx]

            # Compute the contact Jacobian manually.
            normal_dir = contact.frame[:3]
            dist = contact.dist
            fromto = np.empty(6, dtype=np.float64)
            fromto[3:] = contact.pos - 0.5 * dist * normal_dir
            fromto[:3] = contact.pos + 0.5 * dist * normal_dir
            jac = compute_contact_normal_jacobian(
                model,
                data,
                contact.geom1,
                contact.geom2,
                fromto,
                np.empty(3),
                np.empty((3, nv)),
                np.empty((3, nv)),
            )

            np.testing.assert_allclose(jac, efc_J, atol=1e-7)

    def test_qp_upper_bound_branches_with_active_contact(self):
        """Exercise both upper-bound branches with an active contact.

        We place two touching spheres (hi_bound_dist=0) and use a large detection
        distance so the contact is active. With a positive minimum_distance, h equals
        the bound_relaxation (else branch); with a negative minimum_distance, h exceeds
        the relaxation (if branch). G remains identical across both runs.
        """
        xml = """
        <mujoco>
        <worldbody>
            <body name="left">
            <joint type="free"/>
            <geom name="gL" type="sphere" size="0.01" pos="0 0 0"/>
            </body>
            <body name="right">
            <joint type="free"/>
            <geom name="gR" type="sphere" size="0.01" pos="0.02 0 0"/>
            </body>
        </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        cfg = Configuration(model)
        cfg.update(np.zeros(model.nq))

        detect = 0.5
        dt = 1e-3

        # hi_bound_dist (0) <= minimum_distance_from_collisions (positive).
        limit_else = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["gL"], ["gR"])],
            collision_detection_distance=detect,
            minimum_distance_from_collisions=5e-4,  # 0 <= 5e-4 -> else
            bound_relaxation=1e-4,
        )
        G_else, h_else = limit_else.compute_qp_inequalities(cfg, dt)
        assert G_else is not None and h_else is not None

        # All rows that correspond to the active pair should be exactly the relaxation.
        self.assertGreater(h_else.size, 0)
        self.assertTrue(np.any(np.isfinite(h_else)))
        self.assertTrue(np.all(h_else[np.isfinite(h_else)] >= 1e-4 - 1e-12))

        # hi_bound_dist (0) > minimum_distance_from_collisions (negative).
        limit_if = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["gL"], ["gR"])],
            collision_detection_distance=detect,
            minimum_distance_from_collisions=-5e-3,  # 0 > -5e-3 -> if
            bound_relaxation=1e-4,
            gain=0.85,
        )
        G_if, h_if = limit_if.compute_qp_inequalities(cfg, dt)
        assert G_if is not None and h_if is not None

        # h should be strictly larger than relaxation now (gain*dist/dt + relaxation)
        self.assertGreater(h_if.size, 0)
        self.assertTrue(np.any(h_if[np.isfinite(h_if)] > 1e-4 + 1e-12))

        # G should be identical for the same geometry, only h changes.
        self.assertEqual(G_else.shape, G_if.shape)
        np.testing.assert_allclose(G_else, G_if, atol=0, rtol=0)

    def test_constraint_sign_logic_across_distances(self):
        """Verify constraint sign logic works correctly for separated, touching, and
        penetrating geoms.

        This test ensures:
        1. G is non-zero when dist=0
        2. G remains consistent across separated/touching/penetrating states
        3. Constraint correctly distinguishes approaching vs separating velocities
        """
        xml = """
        <mujoco>
        <worldbody>
            <body name="sphere1">
                <joint type="free"/>
                <geom name="g1" type="sphere" size="0.05" pos="0 0 0"/>
            </body>
            <body name="sphere2">
                <joint type="free"/>
                <geom name="g2" type="sphere" size="0.05" pos="0 0 0"/>
            </body>
        </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        cfg = Configuration(model)

        limit = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["g1"], ["g2"])],
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.2,
        )

        # Case 1: Separated geoms (dist > 0).
        qpos_separated = np.zeros(model.nq)
        qpos_separated[7:10] = [0.15, 0, 0]  # sphere2 at x=0.15, dist ≈ 0.05m
        cfg.update(qpos_separated)
        G_sep, h_sep = limit.compute_qp_inequalities(cfg, dt=0.01)
        assert G_sep is not None and h_sep is not None

        # Case 2: Touching geoms (dist ≈ 0).
        qpos_touching = np.zeros(model.nq)
        qpos_touching[7:10] = [0.1, 0, 0]  # sphere2 at x=0.1, dist ≈ 0
        cfg.update(qpos_touching)
        G_touch, h_touch = limit.compute_qp_inequalities(cfg, dt=0.01)
        assert G_touch is not None and h_touch is not None

        # Case 3: Penetrating geoms (dist < 0)/
        qpos_penetrating = np.zeros(model.nq)
        qpos_penetrating[7:10] = [0.05, 0, 0]  # sphere2 at x=0.05, dist < 0
        cfg.update(qpos_penetrating)
        G_pen, h_pen = limit.compute_qp_inequalities(cfg, dt=0.01)
        assert G_pen is not None and h_pen is not None

        # Test 1: G should be non-zero in all cases.
        self.assertTrue(np.any(G_sep[0] != 0), "G should be non-zero when separated")
        self.assertTrue(np.any(G_touch[0] != 0), "G should be non-zero when touching")
        self.assertTrue(np.any(G_pen[0] != 0), "G should be non-zero when penetrating")

        # Test 2: G should be identical across all distance states.
        # (sign correction compensates for normal flip).
        np.testing.assert_allclose(G_sep, G_touch, atol=1e-10, rtol=0)
        np.testing.assert_allclose(G_sep, G_pen, atol=1e-10, rtol=0)

        # Test 3: Verify constraint behavior for approaching vs separating velocities.
        # Create approaching velocity (sphere2 moves toward sphere1 in -x direction).
        qd_approaching = np.zeros(model.nv)
        qd_approaching[6] = -1.0  # -x velocity for sphere2

        # Create separating velocity (sphere2 moves away from sphere1 in +x direction).
        qd_separating = np.zeros(model.nv)
        qd_separating[6] = 1.0  # +x velocity for sphere2

        # Constraint is G @ qd <= h.
        # For approaching: G @ qd should be positive (constraint active).
        # For separating: G @ qd should be negative (constraint inactive).
        g_dot_qd_approaching = G_sep[0] @ qd_approaching
        g_dot_qd_separating = G_sep[0] @ qd_separating

        self.assertGreater(
            g_dot_qd_approaching,
            0,
            "G @ qd should be positive for approaching velocity",
        )
        self.assertLess(
            g_dot_qd_separating, 0, "G @ qd should be negative for separating velocity"
        )

        # Test 4: Verify constraint is satisfied for separating but may be active for
        # approaching.
        self.assertLess(
            g_dot_qd_separating,
            h_sep[0],
            "Separating velocity should satisfy constraint",
        )
        # Approaching may or may not satisfy depending on h, but should be measurable.
        self.assertIsNotNone(g_dot_qd_approaching)

    def test_broadphase_matches_unfiltered(self):
        """Broadphase culling yields the same (G, h) as the unfiltered scan."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body><joint type="slide" axis="1 0 0"/>
              <geom name="s0" type="sphere" size=".1" pos="0 0 0"/></body>
            <body><joint type="slide" axis="1 0 0"/>
              <geom name="s1" type="sphere" size=".1" pos="0 .3 0"/></body>
            <body><joint type="slide" axis="1 0 0"/>
              <geom name="s2" type="sphere" size=".1" pos="0 .6 0"/></body>
            <body><joint type="slide" axis="1 0 0"/>
              <geom name="s3" type="sphere" size=".1" pos="0 .9 0"/></body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        configuration = Configuration(model)
        geoms = ["s0", "s1", "s2", "s3"]
        limit = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(geoms, geoms)],
            collision_detection_distance=0.2,
        )
        # Force the broadphase path regardless of the (small) pair count.
        limit.broadphase_min_pairs = 0

        rng = np.random.RandomState(0)
        culled_at_least_once = False
        for _ in range(50):
            configuration.update(rng.uniform(-1.0, 1.0, size=model.nq))

            limit.broadphase = False
            ref = limit.compute_qp_inequalities(configuration, dt=1e-2)
            limit.broadphase = True
            opt = limit.compute_qp_inequalities(configuration, dt=1e-2)

            # array_equal treats inf == inf as equal, which is what we want for h.
            np.testing.assert_array_equal(ref.G, opt.G)
            np.testing.assert_array_equal(ref.h, opt.h)

            survivors = limit._broadphase_survivors(configuration.data)
            culled_at_least_once |= survivors.size < limit.max_num_contacts

        # Ensure the test actually exercised the culling branch (not vacuously equal).
        self.assertTrue(culled_at_least_once)

    def test_broadphase_plane_pair_matches_unfiltered(self):
        """Broadphase plane-geom culling yields the same (G, h) as the scan.

        The sphere is nested two bodies deep so the geom-vs-plane pair survives
        the parent-child filter (a plane lives in the world body, which would
        otherwise be the weld parent of a top-level geom).
        """
        xml_str = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 .1"/>
            <body>
              <joint type="slide" axis="0 0 1"/>
              <geom name="link" type="sphere" size=".05"/>
              <body>
                <joint type="slide" axis="0 0 1"/>
                <geom name="s" type="sphere" size=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        configuration = Configuration(model)
        limit = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(["s"], ["floor"])],
            collision_detection_distance=0.2,
        )
        limit.broadphase_min_pairs = 0
        # Sanity check: the surviving pair is in the plane-geom broadphase group.
        self.assertEqual(limit._pg_idx.size, 1)

        rng = np.random.RandomState(0)
        culled_at_least_once = False
        for _ in range(50):
            configuration.update(rng.uniform(-0.5, 1.0, size=model.nq))

            limit.broadphase = False
            ref = limit.compute_qp_inequalities(configuration, dt=1e-2)
            limit.broadphase = True
            opt = limit.compute_qp_inequalities(configuration, dt=1e-2)

            np.testing.assert_array_equal(ref.G, opt.G)
            np.testing.assert_array_equal(ref.h, opt.h)

            survivors = limit._broadphase_survivors(configuration.data)
            culled_at_least_once |= survivors.size < limit.max_num_contacts

        self.assertTrue(culled_at_least_once)


if __name__ == "__main__":
    absltest.main()
