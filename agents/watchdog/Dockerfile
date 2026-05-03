FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y build-essential

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. Install Watchdog-specific dependencies (This includes the 'docker' SDK)
COPY agents/watchdog/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS prod
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

# 2. Copy only the necessary modules
# We need discovery/src for the BaseAgent, and watchdog/src for the logic
COPY agents/discovery/src/ ./discovery/src/
COPY agents/watchdog/src/ ./watchdog/src/

# 3. Boot the Watchdog
ENTRYPOINT ["python", "watchdog/src/watchdog_agent.py"]