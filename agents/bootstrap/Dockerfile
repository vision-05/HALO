FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY bootstrap/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM builder as test-builder

COPY bootstrap/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

FROM ubuntu/python:3.10-22.04_stable AS base

COPY --from=test-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

FROM base as test

COPY bootstrap/src/ ./bootstrap/src/
COPY bootstrap/test/ ./bootstrap/test/
COPY discovery/src/ ./discovery/src/


ENTRYPOINT ["pytest", "-s", "bootstrap/test/"]

FROM base as prod

RUN adduser --disabled-password --gecos "" agent

COPY src/ ./src/

RUN chown -R agent:agent /bootstrap/src

USER agent

ENTRYPOINT ["python", "bootstrap/src/main.py"]