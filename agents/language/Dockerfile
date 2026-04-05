FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM ubuntu/python:3.10-22.04_stable AS base

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


WORKDIR /app

FROM base as test

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY src/ ./src/
COPY test/ ./test/


ENTRYPOINT ["pytest", "test/"]

FROM base as prod

RUN adduser --disabled-password --gecos "" agent

COPY src/ ./src/

RUN chown -R agent:agent /src

USER agent

ENTRYPOINT ["python", "src/main.py"]
