# Description of the HALO protocol v0.0.1
This document formally describes the initial iteration of the HALO protocol. It is designed to be robust, future-proof and secure. We keep future backwards-compatibility in mind when designing this protocol.
The HALO protocol describes how a set of autonomous agents work together in order to safely, securely and effectively automate smart homes. The central idea is that there is no central brain. Agents can come and go as they please.
Agents are a virtually distributed system, that can be logically dense.

There are 3 generic agents with a special 4th type of agent:
Actuators, Aggregators, Advocators and Semantic agents.

Actuator agents are in charge of communicating with hardware. They exist usually on a 1:1 scale with smart home devices
Advocator agents are in charge of advocating for house occupants. They also exist 1:1 with humans. They will typically exist on edge devices such as smartphones (so advocacy spans into other HALO enabled houses)
Aggregator agents are in charge of fetching information from external sources. They can be spawned in and despawned on request by other agents, and connect to external APIs

Semantic agents act as the primary direct communication channel between occupants and a household. They contain specific modes of communication, eg LLM, TTS, etc, and can directly update the network.

The network usually learns on its own, but the semantic agents allow for fast-forwarding and preference overrides, as well as quick and seamless setup. Semantic agents also have the ability to periodically refresh the distributed state organisation.

## Generic agents


### Discovery and Heartbeat

### Message format

### Response format

### User Permissions

### Authentication


## Advocator agents

## Aggregator agents

## Actuator agents

## Semantic agents



