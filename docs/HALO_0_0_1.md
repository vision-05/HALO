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
Generic agents are built using the ZeroMQ sockets framework. This abstracts away the transport layer level of networking and leaves us with sockets and different network patterns.

The lifecycle of a HALO network is broken down into the following sections:

- Network startup
- Agent startup
- Agent discovery
- Agent verification
- Heartbeat
- Message passing
- Negotiation
- Agent cleanup
- Agent spawning

### Network startup
A new HALO network is initialised. A dedicated HALO initialisation agent is run on a laptop or desktop PC. This generates a UUID for the network, a root ed25519 keypair (for encryption) and an initial Recognised Device Table, which has those two items as the first entry. Any agent that has the public key for the root can be written to the Recognised Device Table, where its own public key and uuid will be written.

The root private key is displayed as a QR code, or as plaintext that is prompted to be saved by the user.

The first user will startup an advocator agent on their phone, which also generates a UUID. You scan the QR code and the initialisation agent uses the key to send update the RDT, the advocator agent is now admin, writing the UUID, public key and privelige level into the RDT's entry for the network.

After this, the agent overwrites the root private key with zeros, and cleans up the rest of its process, despawning.

### Discovery and Heartbeat
We use mDNS via Python Zeroconf to fetch an unused IP from the network and bind the agent's ZeroMQ sockets to the IP address. This is then broadcast on the mDNS network, until an agent discovers it. The agent can then bind a sandbox interface to the unverified new agent, and begin the verification and authentication process.

### Message format
JSON is the initial choice of message format for our network. It is easily serialisable, and human readable for easy accountability and introspection.

A HALO message can contain any number of fields, and will always inlucde a sender identity.

### User Permissions
We will use a trust system for the network. In a network

### Authentication
We use CurveZMQ for a cryptographic handshake for new connections. This is built in functionality, so long as we provide the keys when creating the sockets.

We need to be able to distribute keys to new agents. When a new agent is spun up, it is not known or trusted. It will generate its own public/private key pair. There is a single pair of keys per network. Agents are designed so that their state can be shared between networks, but messages cannot be. So a message from one network to an advocator cannot then cause the advocator to send a message to another network.

When other agents detect the mDNS broadcast of a new agent, one will prompt the household users whether to pair the new agent. If the user confirms, the public key will be sent over and be placed into the trusted key directory. This is stored on every single node in the mesh. Then the contents of the 

## Advocator agents

## Aggregator agents

## Actuator agents

## Semantic agents
Requests trigger RL rewards.


