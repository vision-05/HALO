FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY language/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM builder as test-builder

COPY language/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

FROM ubuntu/python:3.10-22.04_stable AS base

COPY --from=test-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

FROM base as test

COPY langauge/src/ ./language/src/
COPY language/test/ ./language/test/
COPY discovery/src/ ./discovery/src/


ENTRYPOINT ["pytest", "-s", "language/test/"]

FROM base as prod

COPY language/src/ ./language/src/
COPY discovery/src/ ./discovery/src/

ENTRYPOINT ["python", "language/src/main.py"]
