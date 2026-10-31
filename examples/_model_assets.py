"""Load local example XML with robot assets from MuJoCo Menagerie."""

from pathlib import Path
from xml.etree import ElementTree

import mujoco
import mujoco_menagerie as mm


def load_example_spec(robot_name: str, xml_path: Path) -> mujoco.MjSpec:
    """Preserve a local scene and resolve its mesh and texture files.

    Local files take precedence over packaged assets. This keeps custom assets
    and robot variants intact. The returned specification supports model edits
    and attachments before compilation.
    """
    xml_path = xml_path.resolve()
    spec = mujoco.MjSpec.from_file(str(xml_path))
    package_root: Path | None = None
    original_files: dict[Path, str] = {}

    def collect_files(path: Path) -> None:
        for element in ElementTree.parse(path).iter():
            if element.tag == "include":
                filename = element.attrib["file"]
                include_path = xml_path.parent / filename
                if not include_path.is_file():
                    include_path = path.parent / filename
                collect_files(include_path)
            elif element.tag in {"mesh", "texture"}:
                for attribute in (
                    "file",
                    "fileleft",
                    "fileright",
                    "fileup",
                    "filedown",
                    "filefront",
                    "fileback",
                ):
                    filename = element.get(attribute)
                    if filename:
                        original_files[(path.parent / filename).resolve()] = filename

    # MuJoCo can replace a missing included asset with an absolute path relative
    # to the include file. Recover its XML filename before applying asset dirs.
    collect_files(xml_path)

    def resolve_file(filename: str, asset_directory: str) -> str:
        nonlocal package_root
        if not filename:
            return filename
        file_path = Path(filename)
        if file_path.is_absolute() and not file_path.is_file():
            filename = original_files.get(file_path.resolve(), filename)
        local_path = (xml_path.parent / asset_directory / filename).resolve()
        if local_path.is_file():
            return str(local_path)
        relative_path = local_path.relative_to(xml_path.parent)
        if package_root is None:
            package_root = mm.get(robot_name).path().resolve()
        package_path = package_root / relative_path
        if not package_path.is_file():
            raise FileNotFoundError(
                f"Example asset {relative_path} is absent from {xml_path.parent} "
                f"and the {robot_name} package model."
            )
        return str(package_path)

    # Parsed asset paths need explicit replacement before compilation.
    for mesh in spec.meshes:
        mesh.file = resolve_file(mesh.file, spec.meshdir)
    for texture in spec.textures:
        texture.file = resolve_file(texture.file, spec.texturedir)
        texture.cubefiles = [
            resolve_file(filename, spec.texturedir) for filename in texture.cubefiles
        ]
    return spec
