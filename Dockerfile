# Pinned through Artifact Registry's Docker Hub remote cache so CI does not
# depend on Docker Hub availability for this base image.
ARG PYTHON_BASE_IMAGE=us-west1-docker.pkg.dev/planar-ray-472112-e8/docker-hub-cache/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app
COPY pyproject.toml ./
COPY requirements.lock requirements-voice.lock ./
ARG INSTALL_VOICE=false
RUN if [ "$INSTALL_VOICE" = "true" ]; then \
        pip install --no-cache-dir -r requirements-voice.lock; \
    else \
        pip install --no-cache-dir -r requirements.lock; \
    fi
COPY app ./app
RUN pip install --no-cache-dir --no-deps .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Cloud Run provides PORT; local containers default to 8000.
CMD ["python", "-m", "app.entrypoint"]
