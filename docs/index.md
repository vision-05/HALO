# HALO

Holistic Agentic Living Orchestration

A secure, decentralised, privacy oriented agentic system for home automation.

## Instructions
### Running the agents

Make sure to install docker:

=== "Windows"

    ```bash
    winget install -e --id Docker.DockerDesktop
    ```

=== "MacOS"

    ```bash
    brew install docker docker-compose
    ```

If on mac/linux you might need to prepend `sudo` to your command

=== "bootstrap"

    ```bash
    docker compose up bootstrap
    ```

    Or

    ```bash
    docker compose up bootstrap-test
    ```

=== "language"

    ```bash
    docker compose up language
    ```

    Or

    ```bash
    docker compose up language-test
    ```

=== "Others"

    ```bash
    docker compose up generic
    ```

    Or

    ```bash
    docker compose up generic-test
    ```