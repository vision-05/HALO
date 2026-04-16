FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

# The builder stage just needs what is required to compile Python packages
RUN apt-get update && \
    apt-get install -y build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY generic/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM builder as test-builder
COPY generic/requirements-test.txt .
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


FROM base as test
COPY --from=test-builder /opt/venv /opt/venv
COPY generic/src/ ./generic/src/
COPY generic/test/ ./generic/test/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["pytest", "-s", "generic/test/"]


FROM base as prod
COPY generic/src/ ./generic/src/
COPY discovery/src/ ./discovery/src/
ENTRYPOINT ["python", "generic/src/main.py"]

FROM base as tv

COPY generic/src/ ./generic/src/
COPY discovery/src/ ./discovery/src/

ENTRYPOINT ["python", "generic/src/tcltv.py"]
