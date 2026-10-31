"""Check local scene assets, package assets, and specification attachments."""

import base64
import tempfile
from pathlib import Path
from unittest import mock

import mujoco
from absl.testing import absltest
from examples._model_assets import load_example_spec

_MESH = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMISKkA"
    "AAI0AS3HG8jrAAAAAElFTkSuQmCC"
)


class TestModelAssets(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.local = self.root / "local"
        self.package = self.root / "package"
        self.local.mkdir()
        self.package.mkdir()

    def write(self, path: Path, content: str | bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def test_nested_includes_and_asset_directories(self):
        self.write(
            self.local / "scene.xml",
            '<mujoco><include file="nested/robot.xml"/></mujoco>',
        )
        self.write(
            self.local / "nested/robot.xml",
            """<mujoco>
              <compiler assetdir="assets"/>
              <asset>
                <mesh name="left" file="left/part.obj"/>
                <mesh name="right" file="right/part.obj"/>
                <texture name="custom" type="2d" file="custom.png"/>
                <texture name="generated" type="2d" builtin="checker"
                         width="8" height="8"/>
              </asset>
              <worldbody><body name="robot">
                <geom type="mesh" mesh="left"/>
                <geom type="mesh" mesh="right" pos="2 0 0"/>
              </body></worldbody>
            </mujoco>""",
        )
        self.write(self.local / "assets/left/part.obj", _MESH)
        self.write(self.package / "assets/right/part.obj", _MESH)
        self.write(self.local / "assets/custom.png", _PNG)
        with mock.patch("examples._model_assets.mm.get") as get:
            get.return_value.path.return_value = self.package
            spec = load_example_spec("test_robot", self.local / "scene.xml")
        get.assert_called_once_with("test_robot")
        self.assertEqual(
            spec.mesh("left").file, str(self.local / "assets/left/part.obj")
        )
        self.assertEqual(
            spec.mesh("right").file, str(self.package / "assets/right/part.obj")
        )
        self.assertEqual(
            spec.texture("custom").file, str(self.local / "assets/custom.png")
        )
        self.assertEqual(spec.texture("generated").file, "")
        model = spec.compile()
        self.assertEqual(model.nmesh, 2)
        self.assertEqual(model.ntex, 2)
        spec.body("robot").add_site(name="attachment")
        root = mujoco.MjSpec()
        root.worldbody.add_site(name="mount")
        root.attach(spec, prefix="robot/", site=root.site("mount"))
        self.assertEqual(
            root.compile().site("robot/attachment").name, "robot/attachment"
        )

    def test_local_asset_takes_precedence(self):
        self.write(
            self.local / "robot.xml",
            """<mujoco><compiler meshdir="assets"/>
            <asset><mesh name="part" file="part.obj"/></asset>
            <worldbody><geom type="mesh" mesh="part"/></worldbody></mujoco>""",
        )
        self.write(self.local / "assets/part.obj", _MESH)
        self.write(self.package / "assets/part.obj", "different package asset")
        with mock.patch("examples._model_assets.mm.get") as get:
            spec = load_example_spec("test_robot", self.local / "robot.xml")
        get.assert_not_called()
        self.assertEqual(spec.compile().nmesh, 1)

    def test_nested_include_uses_the_main_directory_first(self):
        self.write(
            self.local / "scene.xml",
            '<mujoco><include file="nested/wrapper.xml"/></mujoco>',
        )
        self.write(
            self.local / "nested/wrapper.xml",
            '<mujoco><include file="shared.xml"/></mujoco>',
        )
        self.write(
            self.local / "shared.xml",
            '<mujoco><worldbody><site name="main"/></worldbody></mujoco>',
        )
        self.write(
            self.local / "nested/shared.xml",
            '<mujoco><worldbody><site name="nested"/></worldbody></mujoco>',
        )
        spec = load_example_spec("test_robot", self.local / "scene.xml")
        self.assertEqual(spec.compile().site("main").name, "main")

    def test_packaged_cube_texture_files(self):
        faces = ("left", "right", "up", "down", "front", "back")
        attributes = " ".join(f'file{face}="{face}.png"' for face in faces)
        self.write(
            self.local / "robot.xml",
            f'<mujoco><compiler texturedir="images"/>'
            f'<asset><texture name="cube" type="cube" {attributes}/></asset></mujoco>',
        )
        for face in faces:
            self.write(self.package / f"images/{face}.png", _PNG)
        with mock.patch("examples._model_assets.mm.get") as get:
            get.return_value.path.return_value = self.package
            spec = load_example_spec("test_robot", self.local / "robot.xml")
        for filename in spec.texture("cube").cubefiles:
            self.assertTrue(Path(filename).is_absolute())
            self.assertTrue(Path(filename).is_file())
        self.assertEqual(spec.compile().ntex, 1)

    def test_missing_asset_names_the_model_and_file(self):
        self.write(
            self.local / "robot.xml",
            '<mujoco><asset><mesh file="missing.obj"/></asset></mujoco>',
        )
        with mock.patch("examples._model_assets.mm.get") as get:
            get.return_value.path.return_value = self.package
            with self.assertRaisesRegex(FileNotFoundError, "missing.obj.*test_robot"):
                load_example_spec("test_robot", self.local / "robot.xml")


if __name__ == "__main__":
    absltest.main()
