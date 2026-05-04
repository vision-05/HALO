# SLM Integration

Traditional smart home systems (e.g., Amazon Alexa, Google Nest) rely on Cloud-based architectures where sensitive behavioral data is processed on remote servers. This introduces significant privacy risks, latency issues, and a "single point of failure" if internet connectivity is lost.
For the HALO project, we have transitioned to a Local-First Agentic Architecture. Central to this is the hosting of Small Language Models (SLMs) directly on edge nodes. Unlike Large Language Models (LLMs) that require massive data centers, SLMs are optimized for high-performance reasoning on low-power hardware. In the HALO ecosystem, the SLM acts as a ‘cognitive’ layer for the Advocate Agent, translating complex, human preferences into the rigid mathematical parameters required for peer-to-peer negotiation.
Role of SLMs in HALO

The Small Language Model (SLM) functions as the decentralized intelligence core of the HALO framework, moving beyond the limitations of cloud-dependent, hub-centric automation. By distributing 'cognitive' tasks across the peer-to-peer network, each node gains the autonomy to process local user context without relying on a central server. This distributed approach allows the Advocate, Governor, and Aggregator agents to function as a cohesive, intelligent ecosystem where reasoning is localized and resilient. In this model, the SLM does not act as a remote controller, but as an embedded logic engine that empowers individual agents to navigate complex environmental and personal trade-offs in real-time, ensuring the system remains private, scalable, and entirely self-contained.

## Selecting SLM Model
To identify the optimal model for the Advocate Agent, three types of SLM models were evaluated with regards to how well they might perform on ARM-based edge hardware (Raspberry Pi 5) and their ability to follow complex instructions.

1. **Llama 3.2 (1B/3B):** Strong general reasoning but occasionally prone to "conversational filler" that can interfere with structured data protocols. Could lead to negotiation oscillation if too chatty.

2. **Phi-3.5 Mini (3.8B):** High logical reasoning scores but proved too computationally heavy for the Raspberry Pi, leading to thermal throttling and high latency.

3. **Qwen 2.5 (1.5B/7B):** Specifically optimized for coding, mathematics, and structured JSON output.

Out of these 3 considered options, Qwen 2.5 was selected as the primary engine for the HALO project due to three critical factors:

- **Protocol Reliability:** Our architecture requires the agent to output a precise weight matrix for the ADMM (Alternating Direction Method of Multipliers) negotiation script. Qwen shows the highest success rate in maintaining JSON schema integrity.

- **Parameter Efficiency:** The 1.5B variant offers a good balance between intelligence and speed, allowing for near-instant "Policy Generation" on the Raspberry Pi 5.

- **Instruction Following:** It excels at multi-constraint reasoning (e.g., "Prioritize user comfort due to illness, but acknowledge the high carbon intensity signal from the Aggregator").

## Policy-to-Control Decoupling

To ensure system stability, we have decoupled "Reasoning" from "Negotiation." Relying just on natural language negotiation can lead to instability and oscillation. By adding the ADMM we add a layer of mathematical reasoning and the negotiation becomes a convex optimisation problem that can converge to an optimal solution. 

- **The SLM (Policy Maker):** The Qwen model analyzes user context and outputs a Dual-Output Schema. This includes a JSON block of mathematical weights and a Natural Language explanation for the user.

- **ADMM Script (The Negotiator):** A lightweight Python layer takes these weights and performs the mathematical negotiation.

This approach solves the risk of oscillating negotiation, identified in our initial risks. By using maths for the negotiation we guarantee convergence, while the SLM provides the "intelligence" and "human interface."

## Experimental Environments: Simulation + Mock Hardware

**Simulation**

For large-scale stress testing, we utilize a Dockerized Agentic Hub. In this environment, we can deploy Qwen 2.5 7B (MAYBE). This allows us to simulate the "Upper Limit" of the system’s intelligence. The simulation environment will use the same Python negotiation logic as the mock hardware, ensuring that the "Bridge" between simulation and reality is purely a matter of computational scale, not architectural change. 

**Mock Hardware Setup**

The physical prototype will utilize a Raspberry Pi 5 (8GB RAM) (MAYBE) hosting Qwen 2.5 1.5B via a 4-bit quantized GGUF format. This setup proves that a local-first, brokerless P2P network is achievable on current consumer-grade edge hardware. Communication between the Advocate, Governor, and Aggregator agents is handled via ZeroMQ. Unlike traditional broker-based systems (e.g., MQTT), ZeroMQ allows for direct, high-performance messaging between autonomous agents. This eliminates the central hub as a bottleneck, ensures local-first data privacy.

## Scalability 

A consideration when choosing the SLM model is the future possibility of scaling HALO to production level, and how close the experimental environments should be to the potential scaled up version to preserve architectural authenticity. Advantages of our chosen model for scaling:

- **Vertical Scaling:** As hardware evolves, the system can swap the 1.5B model for a 7B or 14B model without changing a single line of the P2P protocol, or python scripts.

- **Architectural Symmetry:** Because the Qwen 2.5 family shares a unified tokenizer and training foundation across all sizes, the transition from the 1.5B model (used for mock hardware) to the 7B or 32B models (used in simulation) is seamless. This ensures that the reasoning logic and prompt engineering remain consistent, regardless of the hardware's computational power.

- **Quantization Resilience:** The Qwen family maintains high "reasoning-to-size" density. Even when compressed into 4-bit GGUF formats to fit the Raspberry Pi’s memory constraints, it retains the logical precision required to generate valid JSON weights, ensuring the prototype is a high-fidelity representation of a full-scale system.

- **Native Agentic Alignment:** The model is specifically optimized for function-calling and structured data output. This specialized training ensures that as the network scales to manage more devices, the "Advocator" can interact with the P2P network and Matter API with deterministic reliability, minimizing the risk of protocol failure.