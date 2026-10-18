"""The lab compose file stays consistent with the mock SOAR that drives it."""

from __future__ import annotations

from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1] / "deploy" / "lab"


def compose() -> dict:
    return yaml.safe_load((LAB / "docker-compose.yml").read_text())


def test_the_lab_network_has_no_egress() -> None:
    c = compose()
    assert c["networks"]["lab"]["internal"] is True
    for name, svc in c["services"].items():
        assert "ports" not in svc, f"{name}: ports are not published on an internal network"
    for name, svc in c["services"].items():
        assert svc.get("networks") == ["lab"], f"{name} must sit only on the internal lab network"


def test_mock_soar_is_opt_in_and_shares_the_jef_image() -> None:
    c = compose()
    soar, jef = c["services"]["mock-soar"], c["services"]["jef"]
    assert soar["profiles"] == ["soar"], "plain `up` must not start the driver"
    assert soar["image"] == jef["image"], (
        "the driver signs grants with jef_scene from the jef image"
    )


def test_driver_secrets_match_the_server_placeholders() -> None:
    c = compose()
    jef = c["services"]["jef"]["environment"]
    soar = c["services"]["mock-soar"]["environment"]
    keys = dict(pair.split(":", 1) for pair in jef["JEF_API_KEYS"].split(","))
    assert soar["JEF_API_KEY"] in keys.values()
    assert soar["JEF_GRANT_SECRET"] == jef["JEF_GRANT_SECRET"]
    for value in (soar["JEF_API_KEY"], soar["JEF_GRANT_SECRET"]):
        assert "not" in value and ("lab" in value), "lab placeholders only, never a real secret"


def test_the_decision_store_lives_on_the_volume_the_image_prepares() -> None:
    c = compose()
    jef = c["services"]["jef"]
    assert jef["environment"]["JEF_DECISION_STORE"].startswith("sqlite:////var/lib/jef/")
    assert "decisions:/var/lib/jef" in jef["volumes"]
    dockerfile = (LAB.parents[1] / "Dockerfile").read_text()
    assert "/var/lib/jef" in dockerfile and "chown -R jef:jef /app /var/lib/jef" in dockerfile
