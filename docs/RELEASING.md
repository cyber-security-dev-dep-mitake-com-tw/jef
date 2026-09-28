# Releasing

Tag-driven and lockstep: `v0.2.0` publishes every package at that version.

```bash
python scripts/set-version.py 0.2.0     # writes all 8 packages + pins siblings
git commit -am "chore: 0.2.0" && git tag v0.2.0 && git push --follow-tags
```

A tag ending in `rc1`/`a1`/`b1` goes to **TestPyPI** and npm's `next` tag
instead, so the pipeline can be rehearsed without burning a version number —
neither registry lets you reuse one.

## One-time setup

| | Where | Status |
|---|---|---|
| npm `jef-ai` organization | npmjs.com | done |
| `HF_TOKEN`, `NPM_TOKEN` | repo secrets | done |
| `pypi`, `testpypi` environments | repo settings | done (created via API) |
| **Trusted Publishers** | pypi.org and test.pypi.org | **see below** |

### PyPI Trusted Publishing

Twelve forms: six projects on each of the two registries. Tedious once, then
never again — and it means there is no PyPI token in this repository at all.

**PyPI** → <https://pypi.org/manage/account/publishing/>
**TestPyPI** → <https://test.pypi.org/manage/account/publishing/>

Under *Add a new pending publisher* → **GitHub**, every field is identical
except the project name:

| Field | Value |
|---|---|
| PyPI Project Name | one of the six below |
| Owner | `cyber-security-dev-dep-mitake-com-tw` |
| Repository name | `jef` |
| Workflow name | `release.yml` |
| Environment name | `pypi` on pypi.org · `testpypi` on test.pypi.org |

`scripts/pypi/` has a console script and a bookmarklet that fill everything but
the project name, and pick the environment from the hostname. They do not
submit.

The six project names — note the fifth: the directory is `jef-sdk-python` but
the distribution is **`jef-sdk`**, and PyPI wants the distribution name:

```
jef
jef-core
jef-scene
jef-server
jef-sdk
jef-train
```

Three things that are easy to get wrong:

- **Workflow name is the filename**, `release.yml`, not
  `.github/workflows/release.yml`. This is the most common cause of a first
  publish failing with an unhelpful message.
- **The environment name must match**, and it differs between the two
  registries because the workflow picks `testpypi` for prereleases.
- **A pending publisher does not reserve the name.** PyPI only creates the
  project on first successful publish, so until then someone else can take it.
  All six were free as of 2026-09-27.

## Rehearse first

The point of the rc path is that Trusted Publishing misconfiguration fails at
upload time with a message that names neither the field nor the fix. Find that
out on TestPyPI:

```bash
python scripts/set-version.py 0.2.0rc1
git commit -am "chore: 0.2.0rc1" && git tag v0.2.0rc1 && git push --follow-tags
```

That exercises the whole pipeline — version check, tests, builds, metadata
validation, TestPyPI upload, npm `next`, the multi-arch image and its startup
probe — without touching the real registries' version space.

## What a release does

| Job | |
|---|---|
| `verify` | Tag matches the files, then lint, types, tests and the scene lint. Runs before anything is built: a tag that disagrees with the packages is the one mistake that cannot be undone after upload. |
| `build-python` | Six sdists and six wheels, then `twine check` — a malformed long description is far clearer here than in PyPI's rejection. |
| `pypi` | Trusted Publishing via OIDC. No token. |
| `npm` | `@jef-ai/sdk` and `n8n-nodes-jef`, with provenance. |
| `ghcr` | Multi-arch `linux/amd64,linux/arm64`, then actually runs the pushed image and probes `/healthz`. arm64 matters: the Proxmox and Apple Silicon paths in `Infra/` assume it. |
| `weights` | `head.npz` and `calibration.json` to Hugging Face and the release. Skipped, not failed, if `models/jef-v0/` is absent. |
| `github-release` | Notes and artifacts. |

`pypi` runs before `ghcr` because the image installs from PyPI.

## Afterwards

```bash
pip install jef==0.2.0 && jef models
npm view @jef-ai/sdk version
docker run --rm -p 8080:8080 ghcr.io/cyber-security-dev-dep-mitake-com-tw/jef:0.2.0
```

Until that image exists, every deployment path in `Infra/` and
`deploy/helm/` points at a tag that cannot be pulled.
