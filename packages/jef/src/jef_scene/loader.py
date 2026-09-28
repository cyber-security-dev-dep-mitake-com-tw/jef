"""Load scenes from YAML, failing loudly at load time.

A scene that is wrong should stop a deploy, not an incident. Loading therefore
validates the schema *and* compiles every gate condition, so an unknown name, a
disallowed attribute or a syntax error surfaces here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from jef_core.errors import SceneError

from .schema import Scene

log = logging.getLogger("jef.scene.loader")

__all__ = ["SceneRegistry", "load_scene", "load_scene_text", "load_scenes"]


def _normalise_bool_keys(node: Any) -> Any:
    """Turn YAML's boolean mapping keys back into the strings the contract uses.

    YAML 1.1 resolves a bare ``true:`` or ``false:`` key to a boolean, so the
    natural way to write a noul question --

        criteria:
          true: ...
          false: ...

    -- arrives as ``{True: ..., False: ...}`` and fails validation against a
    contract whose keys are the strings "true" and "false". Requiring scene
    authors to quote them would be a papercut on the single most common question
    type, so the loader normalises instead.
    """
    if isinstance(node, dict):
        out: dict[Any, Any] = {}
        for key, value in node.items():
            if isinstance(key, bool):
                key = "true" if key else "false"
            out[key] = _normalise_bool_keys(value)
        return out
    if isinstance(node, list):
        return [_normalise_bool_keys(item) for item in node]
    return node


def load_scene_text(text: str, *, source: str = "<string>") -> Scene:
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SceneError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SceneError(f"{source}: a scene must be a mapping, got {type(data).__name__}")
    data = _normalise_bool_keys(data)
    try:
        return Scene.model_validate(data)
    except Exception as exc:
        raise SceneError(f"{source}: {exc}") from exc


def load_scene(path: str | Path) -> Scene:
    p = Path(path)
    return load_scene_text(p.read_text(encoding="utf-8"), source=str(p))


def load_scenes(directory: str | Path) -> Iterator[tuple[Path, Scene]]:
    """Yield every ``*.yaml`` / ``*.yml`` scene under ``directory``."""
    root = Path(directory)
    if not root.is_dir():
        raise SceneError(f"{root}: not a directory")
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        yield path, load_scene(path)


class SceneRegistry:
    """Named scenes, compiled and ready to run."""

    def __init__(self) -> None:
        self._scenes: dict[str, Scene] = {}

    def add(self, scene: Scene, *, compile_with: Any = None) -> None:
        if scene.scene in self._scenes:
            existing = self._scenes[scene.scene]
            raise SceneError(
                f"duplicate scene id {scene.scene!r} (already loaded at version {existing.version})"
            )
        if compile_with is not None:
            compile_with.compile(scene)
        self._scenes[scene.scene] = scene

    def load_dir(self, directory: str | Path, *, compile_with: Any = None) -> int:
        count = 0
        for path, scene in load_scenes(directory):
            log.info("loaded scene %s v%d from %s", scene.scene, scene.version, path)
            self.add(scene, compile_with=compile_with)
            count += 1
        return count

    def get(self, name: str) -> Scene:
        try:
            return self._scenes[name]
        except KeyError:
            known = ", ".join(sorted(self._scenes)) or "(none loaded)"
            raise SceneError(f"unknown scene {name!r}; loaded: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._scenes)

    def __len__(self) -> int:
        return len(self._scenes)

    def __contains__(self, name: object) -> bool:
        return name in self._scenes
