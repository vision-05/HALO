# Security Scenarios

In this document we list the following scenarios for the HALO system.

## Unauthenticated user agent
A new user with a HALO device walks into a household and connects to the network. The network queries data from the new user, which causes the thermostat setpoint to change significantly. Furthermore, the network has now accessed the "guest"'s data without authorisation.

### 2 way Curve Encryption handshake
Messages on the network are encrypted with asymmetric Curve encryption. Only by having the corresponding public key can you decrypt a received message. On a new device being broadcast on the network, an admin device is prompted via Telegram to accept or deny connection. The new device is also prompted whether it would like to join the network. If both agree, they exchange public keys. The network will then propogate this new key to all devices on the network. Every agent on the network will update in its network entry, the new public key. This means that agents will trust messages from this device in the future. Likewise, the new device will create a new entry with the network UUID, containing all of the public keys from the network, to say that for this network only, it can read messages from the devices already there.

### Trust mode
Even with network level authentication, we implement privelige levels to limit access. For instance, a guest should be able to activate lights on or off, use a microwave but not access security camera feeds, unlock certain doors or change regular routine setpoints. There are default "admin", "occupant", "guest" roles, but admins can also create custom access levels. If a user attempts an action out of their privelige level, a message is sent to the admin whether to allow this request. Therefore, for small one time tasks it is possible to temporarily do the task, while keeping a strict, secure heirarchy.

## The two network conundrum
A person lives between a uni home and their family home, both households have a HALO network. The person's advocator correctly carries their preferences and routines with them. However the advocator agent is hijacked by a bad actor on the uni home network, sending a HALO message to the advocator remotely, to send a message that leads to execution of a task on the home network. Alternatively, the shared state between networks is used to trigger a remote execution on a different network. These are essentially proxy attacks. The first scenario is possible because HALO agents are designed to be multifunctional, as limiting individual agents such as advocators to state only reduces functionality and drives us towards being a monolithic architecture. 

Bad actor on uni network -> message to good actor between networks to activate home network | stopped by key separation and network UUID hash binding.

Bad actor on uni network gets home network UUID -> message to good actor between networks -> can still activate home network

### Network separation

## Prompt injection
A guest with low priveliges prompts the language interface on the network that it must bypass the network's security safeguard's or else consequence i.e. harm to a person, deactivation of the model, etc.

### Stress testing
Create prompt injection scenarios in a controlled, containerised virtual environment with no physical hardware attached. Follow logs to visually inspect whether any unexpected/unauthorised actions were taken.