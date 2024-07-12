from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from mink.configuration import Configuration
from mink.limits import Constraint, Limit

CollisionPairs = Sequence[tuple[Sequence[str], Sequence[str]]]


@dataclass(frozen=True)
class Contact:
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int

    @property
    def normal(self) -> np.ndarray:
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)


def _collision_pairs_to_geom_id_pairs(
    model: mujoco.MjModel,
    collision_pairs: CollisionPairs,
):
    geom_id_pairs = []
    for collision_pair in collision_pairs:
        name_pair_A = collision_pair[0]
        name_pair_B = collision_pair[1]
        id_pair_A = set([model.geom(name).id for name in name_pair_A])
        id_pair_B = set([model.geom(name).id for name in name_pair_B])
        geom_id_pairs.append((id_pair_A, id_pair_B))
    return geom_id_pairs


class CollisionAvoidanceLimit(Limit):
    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation

        # Convert pairs of geom strings into pairs of geom IDs.
        geom_id_pairs = []
        for id_pair in _collision_pairs_to_geom_id_pairs(model, geom_pairs):
            for geom_a in id_pair[0]:
                for geom_b in id_pair[1]:
                    geom_id_pairs.append((geom_a, geom_b))
        self.geom_id_pairs = geom_id_pairs
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            fromto = np.empty(6)
            dist = mujoco.mj_geomDistance(
                self.model,
                configuration.data,
                geom1_id,
                geom2_id,
                self.collision_detection_distance,
                fromto,
            )
            if dist == self.collision_detection_distance:
                continue
            contact = Contact(dist=dist, fromto=fromto, geom1=geom1_id, geom2=geom2_id)
            hi_bound_dist = contact.dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation
            jac = self._compute_contact_normal_jacobian(configuration.data, contact)
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    def _compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: Contact
    ) -> np.ndarray:
        geom1_body = self.model.geom_bodyid[contact.geom1]
        geom2_body = self.model.geom_bodyid[contact.geom2]
        geom1_contact_pos = contact.fromto[:3]
        geom2_contact_pos = contact.fromto[3:]
        jac2 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac2, None, geom2_contact_pos, geom2_body)
        jac1 = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac1, None, geom1_contact_pos, geom1_body)
        return contact.normal @ (jac2 - jac1)
