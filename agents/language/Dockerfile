FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY language/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


FROM ubuntu/python:3.10-22.04_stable AS base

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY language/main.py .

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["python", "main.py"]
