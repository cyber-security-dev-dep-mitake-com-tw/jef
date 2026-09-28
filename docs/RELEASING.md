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

Six forms: three projects on each of the two registries. Tedious once, then
never again — and it means there is no PyPI token in this repository at all.

**PyPI** → <https://pypi.org/manage/account/publishing/>
**TestPyPI** → <https://test.pypi.org/manage/account/publishing/>

Under *Add a new pending publisher* → **GitHub**, every field is identical
except the project name:

| Field | Value |
|---|---|
| PyPI Project Name | one of the three below |
| Owner | `cyber-security-dev-dep-mitake-com-tw` |
| Repository name | `jef` |
| Workflow name | `release.yml` |
| Environment name | **`pypi-<project>`** on pypi.org · **`testpypi-<project>`** on test.pypi.org |

### Two limits shaped this, and both fail confusingly

Neither is documented where you would look for it, and both come back as a
bare `#errors` anchor on the form.

**1. The environment must differ per project.** PyPI enforces uniqueness on
`(owner, repository, workflow, environment)` — *not* on the project name. A
monorepo publishing several projects from one workflow can register exactly one
pending publisher under a shared environment; the rest are rejected as
duplicates ([warehouse#16920](https://github.com/pypi/warehouse/issues/16920)).

**2. Three pending publishers per account, total.**

```python
# warehouse/accounts/views.py
# we limit users to no more than 3 pending publishers at once.
if len(self.request.user.pending_oidc_publishers) >= 3:
```

The cap counts only *pending* ones: publishing converts a pending publisher
into an ordinary one and frees the slot. So six projects would have meant two
release waves.

They do not, because the second limit prompted a question worth asking anyway:
`jef-core`, `jef-scene` and `jef-sdk` were never installable apart. Each
depended on the one below it and `jef` pulled in all three, so the split bought
three extra PyPI pages and no user any choice. They are now four modules in the
`jef` distribution — `jef_core`, `jef_scene`, `jef_sdk`, `jef_cli` — and the
import paths are unchanged. What is left matches the real dependency
boundaries: light, `+fastapi`, `+torch`.

| Project | Modules | pypi.org | test.pypi.org |
|---|---|---|---|
| `jef` | `jef_core`, `jef_scene`, `jef_sdk`, `jef_cli` | `pypi-jef` | `testpypi-jef` |
| `jef-server` | `jef_server` | `pypi-jef-server` | `testpypi-jef-server` |
| `jef-train` | `jef_train` | `pypi-jef-train` | `testpypi-jef-train` |

The release workflow publishes each project in its own matrix job under the
matching environment, and the six GitHub environments already exist. The
filling script derives the name from the project and the hostname, so you do
not have to keep the table in your head.

`scripts/pypi/fill-trusted-publisher.js` fills everything but the project name
and picks the environment from the hostname. It does not submit.

Three things that are easy to get wrong:

- **Workflow name is the filename**, `release.yml`, not
  `.github/workflows/release.yml`. This is the most common cause of a first
  publish failing with an unhelpful message.
- **The environment name must match**, and it differs between the two
  registries because the workflow picks `testpypi` for prereleases.
- **A pending publisher does not reserve the name.** PyPI only creates the
  project on first successful publish, so until then someone else can take it.
  All three were free as of 2026-09-28, and pending publishers for all three
  are registered on both registries as of the same day.

## Rehearse first

The point of the rc path is that Trusted Publishing misconfiguration fails at
upload time with a message that names neither the field nor the fix. Find that
out on TestPyPI:

```bash
python scripts/set-version.py 0.2.0rc1
git commit -am "chore: 0.2.0rc1" && git tag v0.2.0rc1 && git push --follow-tags
```

### The npm version is spelled differently, on purpose

`0.2.0rc1` is PEP 440. npm rejects it under strict semver, and its *loose*
parser reads it as `0.2.0-rc1` — which the publish path then normalises, so
the registry would hold a third spelling matching neither the tag nor the
pyproject files. Nothing in CI would catch it: `--check` reads the files, and
the files would say what we wrote.

So `set-version.py` translates when it writes `package.json`:

| tag | PyPI | npm |
|---|---|---|
| `v0.2.0` | `0.2.0` | `0.2.0` |
| `v0.2.0rc1` | `0.2.0rc1` | `0.2.0-rc.1` |
| `v0.2.0a1` | `0.2.0a1` | `0.2.0-alpha.1` |
| `v0.2.0b2` | `0.2.0b2` | `0.2.0-beta.2` |

The dot in `-rc.1` matters: semver compares dot-separated numeric identifiers
numerically, so `rc.9 < rc.10`, while `rc9` and `rc10` are single alphanumeric
identifiers compared as strings — which puts the tenth candidate *before* the
ninth.

That exercises the whole pipeline — version check, tests, builds, metadata
validation, TestPyPI upload, npm `next`, the multi-arch image and its startup
probe — without touching the real registries' version space.

### What the first rehearsal found

`v0.2.0rc1` failed in four independent places, none of which any earlier check
could have caught. Recorded because three of the four were mine and the fourth
is a setting only you can change.

| Job | Failure | Cause |
|---|---|---|
| `build-python` | `InvalidDistribution: Unknown distribution format: 'jef'` | `twine check dist/*` handed twine a *directory*. The distributions build into `dist/<project>/` so each publish job can point at one without globbing; the check needed `dist/*/*`. |
| `ghcr` | `manifest unknown` on the startup probe | `type=semver,pattern={{version}}` runs the tag through a strict semver parser. `0.2.0rc1` is PEP 440, so it emitted **no tag at all** — silently — and the image went up as `sha-<short>` only. Now `type=raw`. |
| `weights` | `ModuleNotFoundError: jef_train` | `uv run` at the workspace root syncs the root project, a virtual package with no members. Two minutes of installing, then nothing importable. Now `--no-project` with `PYTHONPATH`, which is cheaper anyway — publishing 400KB does not need torch. |
| `npm` | `npm error code EOTP` | **Yours.** `NPM_TOKEN` lacks *Bypass 2FA* — see below. |

### What the second rehearsal found

`v0.2.0rc2`: `build-python` and `ghcr` passed. Two more, both real:

| Job | Failure | Cause |
|---|---|---|
| `smoke` | `ConnectError: Connection refused` | The CLI defaults to a server on `localhost:8080`, and the job had none — so `jef models` was never exercising the installed package at all. It now passes `--local`. My local check of the same command had silently talked to a leftover server from the Robot suite, which is exactly the trap this job exists to close. The CLI now also answers a refused connection with the address and the three ways out, instead of a bare traceback. |
| `weights` | `403 Forbidden` on `api/repos/create` | **Not a token problem — the namespace was wrong.** The workflow targeted `dennislee928`, an empty Hugging Face account; `HF_TOKEN` belongs to `dennislee928tw`, which is the account that actually holds this project's models. No token permission could have fixed that. Retargeted to `dennislee928tw/jef-v0` (created 2026-09-28, public, Apache-2.0). |

### Verifying a TestPyPI build

TestPyPI is not a usable index on its own: anyone may claim a name there, and
it holds placeholders at absurd versions. Pointing a resolver at both indexes
picks `fastapi==1.0` — a stub that does not build — over the real 0.115. So
take the dependencies from PyPI and only the packages under test from TestPyPI:

```bash
uv venv --python 3.12 /tmp/check
VIRTUAL_ENV=/tmp/check uv pip install --index-url https://pypi.org/simple/ \
  numpy pydantic pyyaml httpx fastapi "uvicorn[standard]" prometheus-client
VIRTUAL_ENV=/tmp/check uv pip install --prerelease allow --no-deps \
  --index-url https://test.pypi.org/simple/ "jef==X" "jef-server==X"

/tmp/check/bin/jef ask "..." --local --yes-no "是否緊急？"
```

`v0.2.0rc3` passes this: four modules import, the server factory loads, and the
CLI returns a distribution.

To fix the Hugging Face token: <https://huggingface.co/settings/tokens> → a
**Write** token, or a fine-grained token with permission to create repos under
your account. The job now calls `whoami` first and stops on a read-only token.

### The npm token needs "Bypass 2FA"

Classic tokens were removed in November 2025, so every npm token is now a
**granular access token** — and those ship with **Bypass 2FA off by default**.
Publishing with one hits `EOTP`, which reads like a missing authenticator code
rather than a token setting. Advice naming a "classic automation token" is
describing a token type that no longer exists.

<https://www.npmjs.com/settings/~/tokens> → *Generate New Token* → **Granular
Access**, scoped to `@jef-ai` and `n8n-nodes-jef`, read **and write**, with
**Bypass 2FA** enabled. Replace the `NPM_TOKEN` secret.

npm removes bypass-2FA direct publishing in **January 2027**. The durable
answer is npm's own trusted publishing (OIDC, no token — the same model as
PyPI), but unlike PyPI it has no pending-publisher flow: the package has to
exist before a trusted publisher can be attached to it. So the first publish
uses the token, and the workflow switches to OIDC afterwards. That also needs
npm CLI ≥ 11.5.1, newer than what Node 22 bundles.

## What a release does

| Job | |
|---|---|
| `verify` | Tag matches the files, then lint, types, tests and the scene lint. Runs before anything is built: a tag that disagrees with the packages is the one mistake that cannot be undone after upload. |
| `build-python` | Three sdists and three wheels, then `twine check` — a malformed long description is far clearer here than in PyPI's rejection. |
| `pypi` | Trusted Publishing via OIDC. No token. |
| `npm` | `@jef-ai/sdk` and `n8n-nodes-jef`, with provenance. |
| `ghcr` | Multi-arch `linux/amd64,linux/arm64`, then actually runs the pushed image and probes `/healthz`. arm64 matters: the Proxmox and Apple Silicon paths in `Infra/` assume it. |
| `weights` | `head.npz` and `calibration.json` to Hugging Face and the release. Skipped, not failed, if `models/jef-v0/` is absent. |
| `github-release` | Notes and artifacts. |

`ghcr` does not wait for `pypi`: the Dockerfile installs from `./packages` in the build context rather than from the index, so a PyPI problem should not hold back the image.

## Afterwards

```bash
pip install jef==0.2.0 && jef models
npm view @jef-ai/sdk version
docker run --rm -p 8080:8080 ghcr.io/cyber-security-dev-dep-mitake-com-tw/jef:0.2.0
```

Until that image exists, every deployment path in `Infra/` and
`deploy/helm/` points at a tag that cannot be pulled.
