# Deploy Using Docker Compose

This guide provides step-by-step instructions to deploy the Smart Intersection Sample Application using Docker Compose. You'll learn how to:
- Start the application and ensure all services are running.
- Verify the deployment to confirm the application is functioning correctly.

Docker Compose simplifies the deployment process by managing multiple containers as a single application. For more details, see [Docker Compose Documentation](https://docs.docker.com/compose/).

---
## Start the Application
1. **Run the Application**:
    - Use Docker Compose to start the application:
        ```bash
        docker compose up -d
        ```
    - Please be aware that the first startup may take longer than usual. Ensure that all services have fully initialized before proceeding.
    - Some containers in the deployment requires network access.
      If you are in a proxy environment, pass the proxy environment variables as follows:
      - For the node-red service, use the following environment configuration in compose.yml:
        ```yaml
        node-red:
          ...
          environment:
            ...
            - http_proxy=http://proxy.example.com:8080
            - https_proxy=http://proxy.example.com:8080
            - no_proxy=localhost,127.0.0.1,broker.scenescape.intel.com,influxdb2
            - HTTP_PROXY=http://proxy.example.com:8080
            - HTTPS_PROXY=http://proxy.example.com:8080
            - NO_PROXY=localhost,127.0.0.1,broker.scenescape.intel.com,influxdb2
        ```
      - For all the other services, provide an empty proxy configuration in compose.yml to avoid inter-container networking issues:
        ```yaml
        <service_name>:
          ...
          environment:
            ...
            - http_proxy=
            - https_proxy=
            - no_proxy=
            - HTTP_PROXY=
            - HTTPS_PROXY=
            - NO_PROXY=
        ```

2. **Verify the Application**:
    - Check that the application is running:
        ```bash
        docker compose ps
        ```

---

## What to Do Next

- **[How to Use the Application](./how-to-use-application.md)**: Verify the application and access its features.
- **[Troubleshooting Docker Deployments](./support.md#troubleshooting-docker-deployments)**: Find detailed steps to resolve common issues during Docker deployments.
- **[Get Started](./get-started.md)**: Ensure you have completed the initial setup steps before proceeding.
