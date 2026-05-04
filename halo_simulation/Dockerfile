FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

# The builder stage just needs what is required to compile Python packages
RUN apt-get update && \
    apt-get install -y build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY halo_simulation/requirements.txt ./halo_simulation/requirements.txt
RUN pip install --no-cache-dir -r halo_simulation/requirements.txt

FROM builder AS test-builder
COPY halo_simulation/requirements-test.txt ./halo_simulation/requirements-test.txt
RUN pip install --no-cache-dir -r halo_simulation/requirements-test.txt

FROM python:3.10-slim AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y adb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

FROM base AS test
COPY --from=test-builder /opt/venv /opt/venv
COPY halo_simulation ./halo_simulation
ENTRYPOINT ["pytest", "-s", "/halo_simulation/tests/"]


FROM base AS prod
COPY halo_simulation ./halo_simulation
COPY agents/discovery/src/ ./discovery/src/
COPY agents/watchdog/src/ ./watchdog/src/
COPY agents/generic/src/ ./generic/src/
