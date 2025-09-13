# Deploy with Helm

Use Helm to deploy Smart Intersection to a Kubernetes cluster. This guide will help you:
- Add the Helm chart repository.
- Configure the Helm chart to match your deployment needs.
- Deploy and verify the application.

Helm simplifies Kubernetes deployments by streamlining configurations and enabling easy scaling and updates. For more details, see [Helm Documentation](https://helm.sh/docs/).


## Prerequisites

Before You Begin, ensure the following:

- **Kubernetes Cluster**: Ensure you have a properly installed and configured Kubernetes cluster.
- **System Requirements**: Verify that your system meets the [minimum requirements](./system-requirements.md).
- **Tools Installed**: Install the required tools:
    - Kubernetes CLI (kubectl)
    - Helm 3 or later

## Steps to Deploy

To deploy the Smart Intersection Sample Application, copy and paste the entire block of commands below into your terminal and run them:

```bash
# change the permissions of the secrets folder
sudo chown -R $USER:$USER chart/files/secrets
sudo chown -R $USER:$USER src/secrets

# Create namespace
kubectl create namespace smart-intersection

# ==========================
# Apply PersistentVolumes
# ==========================
# Ensure the PersistentVolumes (PVs) are created before deploying the application.

kubectl apply -n smart-intersection -f ./chart/templates/grafana/pv.yaml  # PV for Grafana
kubectl apply -n smart-intersection -f ./chart/templates/influxdb/pv.yaml  # PV for InfluxDB
kubectl apply -n smart-intersection -f ./chart/templates/pgserver/pv.yaml  # PV for pgserver


# Install the chart with secrets injected via --set
helm upgrade --install smart-intersection ./chart \
  --create-namespace \
  --set grafana.service.type=NodePort \
  -n smart-intersection

# Some containers in the deployment requires network access.
# If you are in a proxy environment, pass the proxy environment variables as follows:
# helm upgrade \
#     --install  smart-intersection ./chart \
#     --create-namespace \
#     --set httpProxy="http://proxy.example.com:8080" \
#     --set httpsProxy="http://proxy.example.com:8080" \
#     --set noProxy="localhost\,127.0.0.1" \
#     --set grafana.service.type=NodePort \
#     -n smart-intersection
```

Next, transfer the videos to DL Streamer Pipeline Server with the following commands:

```bash
# Get pod name
DLS_PS_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-dlstreamer-pipeline-server -o jsonpath="{.items[0].metadata.name}")

# Copy videos to the init container
kubectl cp ./src/dlstreamer-pipeline-server/videos smart-intersection/${DLS_PS_POD}:/data/ -c init-dlstreamer-pipeline-server-videos

# Create .done flag
kubectl -n smart-intersection exec $DLS_PS_POD -c init-dlstreamer-pipeline-server-videos -- touch /data/videos/.done
```

Next, transfer the model to DL Streamer Pipeline Server with the following commands:

```bash
# Get pod name
DLS_PS_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-dlstreamer-pipeline-server -o jsonpath="{.items[0].metadata.name}")

# Copy model to the init container
kubectl cp ./src/dlstreamer-pipeline-server/models/intersection smart-intersection/${DLS_PS_POD}:/data/models -c init-dlstreamer-pipeline-server-models

# Create .done flag
kubectl -n smart-intersection exec $DLS_PS_POD -c init-dlstreamer-pipeline-server-models -- touch /data/models/.done
```

Finally, transfer the data base archive to Scene DB with the following commands:

```bash
# Get pod name
PGSERVER_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-pgserver -o jsonpath="{.items[0].metadata.name}")

# Copy archive to the init container
kubectl cp ./src/webserver/smart-intersection-ri.tar.bz2 smart-intersection/${PGSERVER_POD}:/data/ -c init-smart-intersection-ri

# Create .done flag
kubectl -n smart-intersection exec $PGSERVER_POD -c init-smart-intersection-ri -- touch /data/.done
```

## Access Application Services

Use `kubectl port-forward` to access the application services on <protocol>://localhost:<service-port>
For more available options, see [kubectl port-forward options](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/#options)


### Access the Application UI

```bash
WEB_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-web -o jsonpath="{.items[0].metadata.name}")
sudo -E kubectl -n smart-intersection port-forward $WEB_POD 443:443
```

### Access the Grafana UI

```bash
GRAFANA_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-grafana -o jsonpath="{.items[0].metadata.name}")
kubectl -n smart-intersection port-forward $GRAFANA_POD 3000:3000
```

### Access the InfluxDB UI

```bash
INFLUX_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-influxdb -o jsonpath="{.items[0].metadata.name}")
kubectl -n smart-intersection port-forward $INFLUX_POD 8086:8086
```

### Access the NodeRED UI

```bash
NODE_RED_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-nodered -o jsonpath="{.items[0].metadata.name}")
kubectl -n smart-intersection port-forward $NODE_RED_POD 1880:1880
```

### Access the DL Streamer Pipeline Server

```bash
DLS_PS_POD=$(kubectl get pods -n smart-intersection -l app=smart-intersection-dlstreamer-pipeline-server -o jsonpath="{.items[0].metadata.name}")
kubectl -n smart-intersection port-forward $DLS_PS_POD 8080:8080
kubectl -n smart-intersection port-forward $DLS_PS_POD 8555:8555
```

## How to Uninstall the Application

To uninstall the application, run the following command:

```bash
helm uninstall smart-intersection -n smart-intersection
```

## How to Delete the Namespace

To delete the namespace and all resources within it, run the following command:

```bash
kubectl delete namespace smart-intersection
```

## What to Do Next

- **[How to Use the Application](./how-to-use-application.md)**: Verify the application and access its features.
- **[Troubleshooting Helm Deployments](./support.md#troubleshooting-helm-deployments)**: Consolidated troubleshooting steps for resolving issues during Helm deployments.
- **[Get Started](./get-started.md)**: Ensure you have completed the initial setup steps before proceeding.

## Supporting Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Helm Documentation](https://helm.sh/docs/)
