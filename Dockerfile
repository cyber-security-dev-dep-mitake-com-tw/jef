# JEF server image.
#
# Target: a CPU-only Proxmox VM. There is deliberately no CUDA base image and no
# GPU path -- the entire design (frozen backbone, head-only training, encoder
# scale) exists so this runs on commodity vCPU.

# Both stages share one Docker Hub base. Deliberately not ghcr.io: egress to it
# is commonly blocked in the enterprise networks this is meant to run in, and a
# build that only works on an unrestricted laptop is not much of a build.
FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock* ./
COPY packages/ ./packages/

# torch is an optional extra. Images that need the real backbone build with
# --build-arg JEF_EXTRAS=torch; the default image stays small and offline-capable.
ARG JEF_EXTRAS=""
RUN uv venv /opt/venv \
 # --no-editable is load-bearing: the root pyproject declares a uv workspace,
 # so without it uv installs the workspace members as editable .pth files
 # pointing at /build, which does not exist in the runtime stage.
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-editable \
      ./packages/jef-core${JEF_EXTRAS:+[$JEF_EXTRAS]} \
      ./packages/jef-scene \
      ./packages/jef-server


FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system jef && useradd --system --gid jef --home /app jef

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    JEF_HOST=0.0.0.0 \
    JEF_PORT=8080 \
    # Defaults to the deterministic test backbone so `docker run` works with no
    # network. Real deployments set JEF_BACKBONE to a model id.
    JEF_BACKBONE=hashing

COPY --from=builder /opt/venv /opt/venv

# Fail the build rather than ship an image that starts and immediately dies.
# The source tree is gone by this stage, so this catches exactly the class of
# packaging bug -- editable installs, missing modules -- that a build-stage
# check cannot see.
RUN python -c "import jef_core, jef_scene, jef_server, jef_server.factory; print('import check ok')"

WORKDIR /app

# Scenes ship with the image so a default deployment has something to run, and
# so `docker run` demonstrates the layer that distinguishes JEF. Mount over
# /app/scenes to supply your own.
COPY scenes/ /app/scenes/
ENV JEF_SCENES_DIR=/app/scenes

RUN mkdir -p /app/.cache && chown -R jef:jef /app
USER jef

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=2).status==200 else 1)"

CMD ["python", "-m", "jef_server"]
