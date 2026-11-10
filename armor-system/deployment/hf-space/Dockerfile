# Model 1 v2 inference service, as a Hugging Face Docker Space.
#
# This is the same service as ../model1-serving/, repackaged for a free public
# Space. Two differences, both forced by the platform:
#
#  * The weights are not in the image. The Space is public so the URL works for
#    anyone, but Khushi's trained checkpoints are not published: they sit in a
#    PRIVATE Hugging Face model repo and entrypoint.sh pulls them at start-up
#    with the Space secret HF_TOKEN. service.py then hashes all ten against the
#    frozen test evidence and refuses to start if any is wrong, so a bad or
#    partial download fails immediately instead of serving a different model.
#
#  * It listens on 7860, the Spaces default, and runs as uid 1000, which is what
#    Spaces expects for a writable home directory.
FROM python:3.14-slim

# curl for a local health probe; git is not needed, the tree is copied in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000 and only /home/user and /tmp are reliably
# writable, so the checkpoints and the HF cache go under the home directory.
RUN useradd --create-home --uid 1000 user

WORKDIR /app

# Dependencies first so a code change does not reinstall torch.
#
# From PyTorch's CPU index at the versions requirements-serving.txt pins. The
# default Linux wheel bundles the CUDA runtime, which this service can never
# use, and it took the image from roughly 700 MB to 3.1 GB. A free Space does
# not need to pull two and a half gigabytes of unusable GPU libraries on every
# rebuild.
COPY armor-system/models/model1_v2/serving/requirements-serving.txt /tmp/requirements-serving.txt
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements-serving.txt \
 && pip install --no-cache-dir "huggingface_hub==0.36.0" \
 && python -c "import torch; print('torch', torch.__version__, 'cuda build:', torch.version.cuda)"

# The service imports data/prep, models/model1_transformer and
# evaluation/detection, and reads three committed artefacts, so the whole
# armor-system tree goes in rather than a hand-picked subset that would drift.
COPY armor-system /app/armor-system
COPY fetch_checkpoints.py entrypoint.sh /app/

# .gitattributes pins entrypoint.sh to LF, but a Space can also be edited in the
# HF web editor or assembled on a machine that rewrites line endings. A CRLF
# shebang fails as a bare "not found" that points nowhere useful, so strip it
# here too rather than rely on one layer of protection.
RUN sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && chown -R user:user /app

USER user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    MODEL1_CHECKPOINT_ROOT=/home/user/checkpoints \
    MODEL1_HOST=0.0.0.0 \
    MODEL1_PORT=7860 \
    # Every published latency figure is single-threaded. Leaving this unset lets
    # torch grab every core and makes the numbers incomparable.
    MODEL1_TORCH_THREADS=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail --silent http://localhost:7860/healthz || exit 1

CMD ["/app/entrypoint.sh"]
