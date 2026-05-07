FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

# The builder stage just needs what is required to compile Python packages
RUN apt-get update && \
    apt-get install -y build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY advocator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM builder AS test-builder
COPY advocator/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

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
COPY advocator/src/ ./advocator/src/
COPY advocator/test/ ./advocator/test/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["pytest", "-s", "advocator/test/"]


FROM base AS prod
COPY advocator/src/ ./advocator/src/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["python", "advocator/src/main.py"]