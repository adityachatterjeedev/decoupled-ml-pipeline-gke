# Cloud-Native ML Inference: Decoupled MNIST Pipeline on GKE

*This repository contains a project developed for the **Cloud and Machine Learning** graduate course at **New York University (NYU)**.*

This repository contains the deployment manifests, containerization configurations, and microservice source code used to architect an end-to-end, decoupled Machine Learning as a Service (MLaaS) platform on Google Kubernetes Engine (GKE). 

The system implements a Convolutional Neural Network (CNN) to classify handwritten digits in real-time, completely separating stateful, high-compute training loops from stateless, low-latency web serving layers. Persistent shared network volumes orchestrate the transactional lifecycle, ensuring high availability, operational resilience, and automated self-healing.

---

## Project Objectives

Commercial generative and discriminative AI workloads demand strict isolation between training pipelines and serving endpoints to optimize resource consumption and infrastructure costs. This project demonstrates a production-ready blueprint for MLOps orchestration through four primary structural pillars:

* **Compute and Storage Decoupling:** Isolating ephemeral container filesystems from persistent model states to allow independent scaling profiles for training jobs and inference deployments.
* **Orchestration Resilience:** Utilizing declarative Kubernetes objects to build a self-healing cluster capable of automatic pod replication, lifecycle management, and failover routing.
* **Infrastructure Bottleneck Mitigation:** Troubleshooting and engineering around cloud provider resource limits, including disk performance constraints and network IP exhaustion.
* **Race Condition Protection:** Engineering safety mechanisms within application runtimes to synchronize distributed storage dependencies safely across lifecycle state changes.

## System Architecture and Infrastructure Specifications

The deployment architecture abstracts compute resources away from long-term storage boundaries by structuring the machine learning lifecycle into independent, isolated Kubernetes workloads.

```text
.
├── k8s/
│   ├── pvc.yaml
│   ├── serve-deploy.yaml
│   └── train-job.yaml
├── src/
│   ├── inference/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── templates/
│   │       └── index.html
│   └── training/
│       ├── main.py
│       ├── Dockerfile
│       └── requirements.txt
└── README.md
```

### Core Cluster Blueprint
* **Orchestration Layer:** Google Kubernetes Engine (GKE) Cluster, utilizing `e2-standard-2` compute nodes (2 vCPUs, 8 GB RAM per node).
* **Storage Layer:** A dynamically provisioned GCE Persistent Disk backed by a custom `PersistentVolumeClaim` (PVC), allocated with a `10Gi` storage ceiling.
* **Network Topography:** A single-zone cluster architecture deployed alongside a public-facing Kubernetes `LoadBalancer` Service, exposing standard port `80` globally and routing internal transactions to container port `5000`.

### Workflow Dynamics
1. **The State Extraction Job (`train-job.yaml`):** An isolated, high-compute batch container initializes, mounts the shared `PersistentVolume` with read-write access, trains the PyTorch CNN model over the MNIST dataset, serializes the optimized weights matrix into a state dictionary file (`mnist_cnn.pth`), writes it directly to the network directory mount, and gracefully terminates.
2. **The Serving Deployment (`serve-deploy.yaml`):** Concurrently, stateless frontend web application pods spin up behind the external LoadBalancer. These pods mount the identical `PersistentVolume` with strict `readOnly: true` enforcement to prevent state corruption. An integrated runtime polling mechanism forces the Flask engine to wait until the network mount registers the existence of `mnist_cnn.pth`, at which point it loads the weights into a local CPU execution thread to begin handling real-time browser canvas classification requests.

## Cloud Quota Diagnostics and Distributed System Bottlenecks

Transitioning the pipeline from a local minikube sandbox to a multi-node Google Kubernetes Engine (GKE) enterprise cluster unmasked real-world resource walls and race conditions. Below is the forensic analysis of the infrastructure bottlenecks encountered and the engineering remediations applied.

### 1. The Google Cloud Standard Disk Wall
* **The Bottleneck:** Initial deployments requested a high-performance SSD Persistent Disk. However, standard free-tier or graduate GCP quotas enforce a strict cap on maximum Input/Output Operations Per Second (IOPS) and throughput provisioning thresholds. This restriction caused GKE to freeze during volume attachment, leaving the Persistent Volume (PV) stuck in an indefinite `Provisioning` state.
* **The Remediation:** The storage manifest (`pvc.yaml`) was refactored to explicitly drop premium SSD parameters, falling back to standard persistent block storage (`pd-standard`) with a `10Gi` capacity ceiling. This safely aligned the manifest with baseline public cloud quotas while satisfying the storage demands of the serialized PyTorch weights file.

### 2. Regional IP Address Exhaustion (`IN_USE_ADDRESSES`)
* **The Bottleneck:** Setting up the GKE cluster across an entire multi-zone region automatically provisions a distributed control plane that binds multiple external static IP addresses. This deployment strategy immediately violated the project's maximum regional networking allocation, triggering a `Quota 'IN_USE_ADDRESSES' exceeded` infrastructure rejection.
* **The Remediation:** The deployment topology was altered from a multi-zonal regional footprint into a strict single-zone cluster (`us-central1-a`). This consolidated control-plane allocation to a single node-pool interface, freeing up the network quota capacity required by the public-facing `LoadBalancer` Service.

### 3. File Locking Race Conditions under ReadWriteOnce
* **The Bottleneck:** Standard GCE Persistent Disks are structurally bound to a `ReadWriteOnce` (RWO) access mode, meaning they can only be mounted with write permissions by a single cluster node at a time. During cluster rolling updates or system-initiated pod rescheduling, new inference pods attempted to spin up and bind the filesystem while older pods were still terminating. This created an operational deadlock, trapping new container cycles in a perpetual `ContainerCreating` or `VolumeInUse` status.
* **The Remediation:** The pipeline was completely decoupled at the lifecycle layer. The serving application (`serve-deploy.yaml`) was hardcoded to mount the shared directory with strict `readOnly: true` enforcement. Because read-only claims bypass single-node exclusive write locks, GKE can horizontally replicate and scale the inference layer across multiple nodes simultaneously without triggering storage system deadlocks.

## Container Strategy and Build Automation

To ensure reproducible deployment across nodes, both phases of the lifecycle are completely containerized. The strategy focuses on layer footprint optimization and efficient artifact delivery.

### 1. Training Environment Matrix
The training container leverages a minimized runtime base to isolate the high-compute dependencies needed to perform backpropagation on the MNIST dataset:
* **Base Image Optimization:** Uses a standard `python:3.9-slim` base, stripping away heavy compiler tools such as GCC, Make, and G++ (`build-essential`). Since PyTorch and Torchvision are distributed as pre-compiled wheels (`.whl`), removing explicit build dependencies reduces the image assembly footprint by several hundred megabytes.
* **Layer Caching Optimization:** Manifest fields (`requirements.txt`) are injected and processed via `pip install` before copying the remaining source code. This pattern isolates heavy package installation from localized code edits, ensuring that minor adjustments to the PyTorch network weights or hyperparameters do not invalidate the cached Docker layers during automated continuous integration cycles.

### 2. Inference Engine Micro-Framework
The web serving layer encapsulates lightweight application binaries to keep deployment rollouts fast and responsive:
* **Dependency Lightening:** Unlike the training layer, the inference image strips out `torchvision` entirely. The Flask runtime processes client-side input arrays directly into standard PyTorch float tensors using standard matrix dimension manipulations (`.view(1, 1, 28, 28)`), eliminating nearly 1 GB of unexecuted binary baggage.
* **Web Serving Port Mapping:** The inference micro-framework standardizes on Python 3.9, explicitly exposing container port `5000` to register traffic routed inward by the Kubernetes cluster networking layer.

## Deployment and Execution Guide

Follow these steps to build the container images, push them to the Google Artifact Registry (GAR), and orchestrate the cluster components using `kubectl`.

### 1. Artifact Registry Initialization
Create a secure Docker repository inside your Google Cloud project to host the optimized runtime images:
```bash
gcloud artifacts repositories create ml-k8s-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for decoupled MNIST pipeline"
```
Configure local Docker CLI credentials to authenticate against the remote GCP registry endpoint:
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 2. Container Compilation and Push Operations
Build and tag the training image using the streamlined runtime context:
```bash
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-k8s-repo/mnist-train:v2 ./src/training
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-k8s-repo/mnist-train:v2
```
Build and tag the lightened inference web serving layer:
```bash
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-k8s-repo/mnist-infer:v3 ./src/inference
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-k8s-repo/mnist-infer:v3
```

*Note: Replace `YOUR_PROJECT_ID` with your actual Google Cloud Project ID inside the manifests before applying*

### 3. Kubernetes Orchestration Sequence
Navigate to the root configuration directory and initialize the storage architecture. This establishes the volume layer before pods request a mount:
```bash
kubectl apply -f k8s/pvc.yaml
```
Verify that the volume has successfully bound to standard cloud block storage:
```bash
kubectl get pvc model-pvc
```
Deploy the high-compute training workflow to extract and store the neural network weights:
```bash
kubectl apply -f k8s/train-job.yaml
```
Simultaneously initiate the stateless web-serving layer and its public routing configuration:
```bash
kubectl apply -f k8s/serve-deploy.yaml
```

### 4. Lifecycle Verification and Monitoring
Track the progress of the ephemeral batch container until it reaches a completed status:
```bash
kubectl get jobs -w
kubectl logs -l job-name=mnist-training-job
```
Monitor the web-serving infrastructure as the Flask runtime safely executes its active wait-loop polling sequence:
```bash
kubectl get pods -l app=mnist-infer -w
kubectl logs -l app=mnist-infer
```
Retrieve the public static IP assigned by the cloud platform to connect to the user-facing drawing interface:
```bash
kubectl get service mnist-loadbalancer -w
```

## Operational Verification and Core Conclusions

Migrating this decoupled machine learning workload from a local sandbox to an enterprise Google Kubernetes Engine (GKE) environment validated several critical cloud-native production principles:

### 1. Verification Logs and Telemetry
When executing the pipeline, the distributed components communicate seamlessly across the storage substrate:
* **Training Convergence:** The `mnist-training-job` logs confirm standard dataset extraction, processing the epochs to reach nominal loss criteria before committing the state matrix:
  ```text
  Train Epoch: 14 [54400/60000 (91%)]  Loss: 0.048123
  Test set: Average loss: 0.0271, Accuracy: 9918/10000 (99%)
  Model successfully saved to persistent storage at: /mnt/model/mnist_cnn.pth
  ```
* **Runtime Syncing Loop:** The inference engine logs demonstrate the container synchronization safety framework during initialization, successfully holding process execution until the persistent state maps to the namespace:
  ```text
  Waiting for model file to be created by training job...
  Waiting for model file to be created by training job...
  Model loaded successfully!
  * Running on [http://0.0.0.0:5000](http://0.0.0.0:5000)
  ```
### 2. Takeaways and Engineering Conclusions
* **High Availability Serving Infrastructure:** Through declarative replication limits inside the Kubernetes deployment manifests, the system creates a resilient, self-healing framework. GKE continuously monitors pod health signatures, executing instantaneous container restarts and volume re-attachments during hardware disruption without introducing front-facing user service degradation.
* **Commercial Viability of MLOps Decoupling:** By avoiding monolithic architectural anti-patterns, compute expenses drop significantly. Ephemeral training pods request maximum compute resources only when processing datasets, then spin down completely. Concurrently, lightweight inference containers run on small, inexpensive, stateless instances that horizontally scale out to handle variable consumer traffic. 
* **The Cloud Provider Reality Check:** Building this system highlighted that cloud architecture is as much about understanding resource boundaries as it is about writing clean code. Navigating quota ceilings—such as managing regional dynamic IP address allocations and working around standard block storage throughput limits—provided vital hands-on experience right-sizing production multi-tenant environments.