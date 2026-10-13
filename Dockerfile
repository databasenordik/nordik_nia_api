FROM python:3.12-slim

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
