# Model 1 v2 inference service.
#
# SERVING.md section 2 says a container needs only Python 3.14, the two pinned
# packages, the armor-system tree and the checkpoint directory mounted at
# MODEL1_CHECKPOINT_ROOT. That is what this builds.
#
# The checkpoints are NOT baked into the image. They are 3.5 MB of model weights
# kept outside Git, and the service hashes each one against the frozen test
# evidence at start-up. Copying them into a layer would put weights into
# something we may later push to a registry, and would break the hash check on
# every retrain. They are mounted at run time instead.
FROM python:3.14-slim

# curl is for the container healthcheck; the service itself is standard-library
# HTTP plus torch and numpy.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Non-root. The service reads the repository and the checkpoints and writes
# nothing, so it has no reason to run privileged.
RUN useradd --create-home --uid 10001 serving

WORKDIR /app

# Dependencies first so a repository change does not reinstall torch.
COPY armor-system/models/model1_v2/serving/requirements-serving.txt /tmp/requirements-serving.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serving.txt

# The service imports data/prep, models/model1_transformer and
# evaluation/detection, and reads three committed artefacts, so the whole
# armor-system tree goes in rather than a hand-picked subset that would drift.
COPY armor-system /app/armor-system

USER serving

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL1_CHECKPOINT_ROOT=/checkpoints \
    MODEL1_HOST=0.0.0.0 \
    MODEL1_PORT=8081 \
    # All published latency numbers were produced single-threaded. Leaving this
    # unset lets torch grab every core and makes the figures incomparable.
    MODEL1_TORCH_THREADS=1

EXPOSE 8081

# The service refuses to start if a checkpoint is missing or altered, so a
# container that reaches "healthy" has already verified its weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl --fail --silent http://localhost:8081/healthz || exit 1

CMD ["python", "-I", "/app/armor-system/models/model1_v2/serving/server.py"]
