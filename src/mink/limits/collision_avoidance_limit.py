"""Collision avoidance limit."""

import itertools
from typing import Sequence

import mujoco
import numpy as np

from ..configuration import Configuration, _resolve_frame_id
from .limit import Constraint, Limit

# Type aliases.
Geom = int | str
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]

# Minimum number of geom pairs for the vectorized broadphase to be worthwhile.
# Below this, the per-call NumPy overhead of the broadphase exceeds the cost of
# simply looping over the pairs, so we fall back to the linear scan.
_BROADPHASE_MIN_PAIRS = 16


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1_id: int,
    geom2_id: int,
    fromto: np.ndarray,
    normal: np.ndarray,
    jac1: np.ndarray,
    jac2: np.ndarray,
) -> np.ndarray:
    """Compute the contact normal Jacobian between two geoms."""
    normal[:] = fromto[3:] - fromto[:3]
    mujoco.mju_normalize3(normal)
    geom_bodyid = model.geom_bodyid
    mujoco.mj_jac(model, data, jac2, None, fromto[3:], geom_bodyid[geom2_id])
    mujoco.mj_jac(model, data, jac1, None, fromto[:3], geom_bodyid[geom1_id])
    jac2 -= jac1
    return normal @ jac2


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Returns true if the geoms are part of the same body, or if their bodies are
    welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # body_weldid is the ID of the body's weld.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs in which to perform active collision
            avoidance. A collision pair is defined as a pair of geom groups. A geom
            group is a set of geom names. For each geom pair, the solver will
            attempt to compute joint velocities that avoid collisions between every
            geom in the first geom group with every geom in the second geom group.
            Self collision is achieved by adding a collision pair with the same
            geom group in both pair fields.
        gain: Gain factor in (0, 1] that determines how fast the geoms are
            allowed to move towards each other at each iteration. Smaller values
            are safer but may make the geoms move slower towards each other.
        minimum_distance_from_collisions: The minimum distance to leave between
            any two geoms. A negative distance allows the geoms to penetrate by
            the specified amount.
        collision_detection_distance: The distance between two geoms at which the
            active collision avoidance limit will be active. A large value will
            cause collisions to be detected early, but may incur high computational
            cost. A negative value will cause the geoms to be detected only after
            they penetrate by the specified amount.
        bound_relaxation: An offset on the upper bound of each collision avoidance
            constraint.
        broadphase: If True, skip geom pairs that are provably out of collision
            detection range before the expensive narrow-phase distance query. The
            assembled constraint is identical to the unfiltered computation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
        broadphase: bool = True,
    ):
        """Initialize collision avoidance limit.

        Args:
            model: MuJoCo model.
            geom_pairs: Set of collision pairs in which to perform active collision
                avoidance. A collision pair is defined as a pair of geom groups. A geom
                group is a set of geom names. For each collision pair, the mapper will
                attempt to compute joint velocities that avoid collisions between every
                geom in the first geom group with every geom in the second geom group.
                Self collision is achieved by adding a collision pair with the same
                geom group in both pair fields.
            gain: Gain factor in (0, 1] that determines how fast the geoms are
                allowed to move towards each other at each iteration. Smaller values
                are safer but may make the geoms move slower towards each other.
            minimum_distance_from_collisions: The minimum distance to leave between
                any two geoms. A negative distance allows the geoms to penetrate by
                the specified amount.
            collision_detection_distance: The distance between two geoms at which the
                active collision avoidance limit will be active. A large value will
                cause collisions to be detected early, but may incur high computational
                cost. A negative value will cause the geoms to be detected only after
                they penetrate by the specified amount.
            bound_relaxation: An offset on the upper bound of each collision avoidance
                constraint.
            broadphase: If True (default), cheaply skip geom pairs whose bounding
                spheres (or plane half-spaces) are farther apart than the collision
                detection distance before invoking the expensive narrow-phase
                mj_geomDistance. This is a strict pre-filter: it only discards pairs
                that would not produce a constraint, so the resulting constraint is
                identical to the unfiltered computation.
        """
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)
        self.broadphase = broadphase
        # Tunable per instance; set to 0 to force the broadphase path on any scene.
        self.broadphase_min_pairs = _BROADPHASE_MIN_PAIRS

        self._fromto = np.empty(6)
        self._normal = np.empty(3)
        self._jac1 = np.empty((3, model.nv))
        self._jac2 = np.empty((3, model.nv))

        self._init_broadphase(model)

    def _init_broadphase(self, model: mujoco.MjModel) -> None:
        """Precompute the per-pair data used by the vectorized broadphase.

        Pairs are partitioned once into three groups, mirroring MuJoCo's own
        mj_filterSphere (engine_collision_driver.c):

        Sphere-sphere pairs, where both geoms have a finite bounding sphere
        (rbound > 0), are culled by a sphere-sphere distance test. Plane-geom
        pairs, where one geom is a plane and the other is bounded, are culled by
        the signed distance of the other geom's center to the plane. Everything
        else (e.g. a non-plane geom with rbound == 0, such as a height field) is
        never culled, to stay conservative.

        The stored indices point back into geom_id_pairs so survivors map to the
        correct constraint rows.
        """
        pairs = np.array(self.geom_id_pairs, dtype=int).reshape(-1, 2)
        g1, g2 = pairs[:, 0], pairs[:, 1]
        rbound = model.geom_rbound
        is_plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE

        both_bounded = (rbound[g1] > 0.0) & (rbound[g2] > 0.0)
        # A plane has rbound == 0, so plane pairs never appear in both_bounded.
        plane1 = is_plane[g1] & (rbound[g2] > 0.0)  # g1 is the plane.
        plane2 = is_plane[g2] & (rbound[g1] > 0.0)  # g2 is the plane.
        plane_pair = (plane1 | plane2) & ~both_bounded

        ss = np.where(both_bounded)[0]
        pg = np.where(plane_pair)[0]
        keep = np.where(~both_bounded & ~plane_pair)[0]

        # Sphere-sphere group.
        self._ss_idx = ss
        self._ss_g1 = g1[ss]
        self._ss_g2 = g2[ss]
        self._ss_rsum = rbound[g1[ss]] + rbound[g2[ss]]

        # Plane-geom group: identify the plane geom and the bounded "other" geom.
        plane_is_g1 = plane1[pg]
        self._pg_idx = pg
        self._pg_plane = np.where(plane_is_g1, g1[pg], g2[pg])
        self._pg_other = np.where(plane_is_g1, g2[pg], g1[pg])
        self._pg_rother = rbound[self._pg_other]

        # Always-keep group.
        self._keep_idx = keep

    def _broadphase_survivors(self, data: mujoco.MjData) -> np.ndarray:
        """Return the indices into geom_id_pairs that survive broadphase.

        A pair is discarded only when it is farther apart than the collision
        detection distance; mirrors mj_filterSphere with that distance as margin.
        """
        margin = self.collision_detection_distance
        xpos = data.geom_xpos
        survivors = [self._keep_idx]

        if self._ss_idx.size:
            diff = xpos[self._ss_g1] - xpos[self._ss_g2]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            bound = self._ss_rsum + margin
            survivors.append(self._ss_idx[dist_sq <= bound * bound])

        if self._pg_idx.size:
            # Plane normal is the z-axis of the plane frame: columns 2, 5, 8 of xmat.
            normal = data.geom_xmat[self._pg_plane][:, [2, 5, 8]]
            diff = xpos[self._pg_other] - xpos[self._pg_plane]
            signed_dist = np.einsum("ij,ij->i", normal, diff)
            survivors.append(self._pg_idx[signed_dist <= margin + self._pg_rother])

        # Row order is irrelevant (each survivor writes an independent row), so no
        # sort is needed.
        return np.concatenate(survivors)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        model = self.model
        data = configuration.data
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, model.nv))
        fromto = self._fromto
        normal = self._normal
        jac1 = self._jac1
        jac2 = self._jac2
        distmax = self.collision_detection_distance
        min_dist = self.minimum_distance_from_collisions
        gain = self.gain
        relaxation = self.bound_relaxation
        if self.broadphase and self.max_num_contacts >= self.broadphase_min_pairs:
            indices = self._broadphase_survivors(data)
        else:
            indices = range(self.max_num_contacts)
        for idx in indices:
            geom1_id, geom2_id = self.geom_id_pairs[idx]
            dist = mujoco.mj_geomDistance(
                model, data, geom1_id, geom2_id, distmax, fromto
            )
            if abs(dist - distmax) < 1e-12:
                continue
            row = compute_contact_normal_jacobian(
                model, data, geom1_id, geom2_id, fromto, normal, jac1, jac2
            )
            if dist > min_dist:
                upper_bound[idx] = (gain * (dist - min_dist) / dt) + relaxation
            else:
                upper_bound[idx] = relaxation
            sign = -1.0 if dist >= 0 else 1.0
            coefficient_matrix[idx] = sign * row
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> list[int]:
        """Take a heterogeneous list of geoms (specified via ID or name) and return
        a homogenous list of IDs (int)."""
        return [_resolve_frame_id(self.model, g, "geom") for g in geom_list]

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Returns a set of geom ID pairs for all possible geom-geom collisions.

        The contacts are added based on the following heuristics:
            1) Geoms that are part of the same body or weld are not included.
            2) Geoms where the body of one geom is a parent of the body of the other
                geom are not included.
            3) Geoms that fail the contype-conaffinity check are ignored.

        Note:
            1) If two bodies are kinematically welded together (no joints between them)
                they are considered to be the same body within this function.
        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                weld_body_cond = not _is_welded_together(self.model, geom_a, geom_b)
                parent_child_cond = not _are_geom_bodies_parent_child(
                    self.model, geom_a, geom_b
                )
                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(
                    self.model, geom_a, geom_b
                )
                if weld_body_cond and parent_child_cond and contype_conaffinity_cond:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        # Deduplicate pairs in case of overlapping geom groups.
        return list(set(geom_id_pairs))
