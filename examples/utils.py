"""Convenience methods for working with MjSpec."""

from dataclasses import dataclass
from typing import Any

import mujoco


@dataclass
class Mjcf:
    spec: mujoco.MjSpec = mujoco.MjSpec()

    @staticmethod
    def from_xml_path(xml_path: str) -> "Mjcf":
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        return Mjcf(spec)

    @property
    def name(self):
        return self.spec.modelname

    @name.setter
    def name(self, new_name: str) -> None:
        self.spec.modelname = new_name

    def add_checkered_plane(self):
        text = self.spec.add_texture()
        self._set_kwargs(
            text,
            name="grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.1, 0.15, 0.2),
            width=512,
            height=512,
            mark=mujoco.mjtMark.mjMARK_CROSS,
            markrgb=(0.8, 0.8, 0.8),
        )
        mat = self.spec.add_material()
        self._set_kwargs(
            mat,
            name="grid",
            texture="grid",
            texrepeat=(1, 1),
            texuniform=True,
        )
        geom = self.spec.worldbody.add_geom()
        self._set_kwargs(
            geom,
            name="floor",
            size=(1, 1, 0.01),
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            material="grid",
        )

    def add(
        self,
        dtype,
        parent: Any | None = None,
        **kwargs,
    ):
        parent = parent or self.spec.worldbody
        add_fn = f"add_{dtype}"
        elem = getattr(parent, add_fn)()
        self._set_kwargs(elem, **kwargs)
        return elem

    def compile(self) -> mujoco.MjModel:
        return self.spec.compile()

    # Private methods.

    def _set_kwargs(self, entity: Any, **kwargs):
        for name, value in kwargs.items():
            setattr(entity, name, value)
