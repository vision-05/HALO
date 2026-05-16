FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY agents/rl/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM builder AS test-builder
COPY agents/rl/requirements-test.txt /tmp/requirements-test.txt
RUN pip install --no-cache-dir -r /tmp/requirements-test.txt

FROM python:3.10-slim AS base
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
WORKDIR /app

FROM base AS test
COPY --from=test-builder /opt/venv /opt/venv
COPY RL/ ./RL/
COPY routine_learning/ ./routine_learning/
COPY agents/discovery/src/ ./discovery/src/
COPY agents/rl/src/ ./rl/src/
COPY agents/rl/test/ ./rl/test/
ENTRYPOINT ["pytest", "-s", "rl/test/"]

FROM base AS prod
COPY RL/ ./RL/
COPY routine_learning/ ./routine_learning/
COPY agents/discovery/src/ ./discovery/src/
COPY agents/rl/src/ ./rl/src/
ENTRYPOINT ["python", "rl/src/main.py"]

