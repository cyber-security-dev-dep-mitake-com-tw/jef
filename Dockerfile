# JEF server image.
#
# Target: a CPU-only Proxmox VM. There is deliberately no CUDA base image and no
# GPU path -- the entire design (frozen backbone, head-only training, encoder
# scale) exists so this runs on commodity vCPU.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build
COPY pyproject.toml uv.lock* ./
COPY packages/ ./packages/

# torch is an optional extra. Images that need the real backbone build with
# --build-arg JEF_EXTRAS=torch; the default image stays small and offline-capable.
ARG JEF_EXTRAS=""
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
      ./packages/jef-core${JEF_EXTRAS:+[$JEF_EXTRAS]} \
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

WORKDIR /app
RUN mkdir -p /app/.cache && chown -R jef:jef /app
USER jef

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=2).status==200 else 1)"

CMD ["python", "-m", "jef_server"]
