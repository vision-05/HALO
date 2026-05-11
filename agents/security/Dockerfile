FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY security/requirements.txt .
COPY security/requirements-test.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

FROM builder as test-builder

FROM python:3.10-slim AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1 libgles2 libegl1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=test-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

FROM base as test
COPY security/src/ ./security/src/
COPY security/test/ ./security/test/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["pytest", "-s", "security/test/"]

FROM base as prod
COPY security/src/ ./security/src/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["python", "security/src/suspicion_detection.py"]
