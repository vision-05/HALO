FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY security/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM builder as test-builder

COPY security/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

FROM ubuntu/python:3.10-22.04_stable AS base
ENV DEBIAN_FRONTEND=noninteractive

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
