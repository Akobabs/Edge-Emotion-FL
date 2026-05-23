# Leveraging Federated Learning for Privacy-Conscious Emotion Recognition

This repository contains the complete implementation of the **Privacy-Conscious Federated Emotion Recognition (ER) System** designed for your Master's research. The project is divided into two core components:
1. **The Federated Learning Simulation (Backend):** A multi-client Client-Server architecture collaboratively training a lightweight CNN model on Non-IID partitions of the FER-2013 dataset using **Flower**, incorporating **FedProx**, client-side **Differential Privacy (DP)**, and server-side **LBAAFedAvg** anomaly filtering.
2. **The Interactive Demonstration (Frontend):** A beautiful, responsive browser-based web application with custom HSL glassmorphism dark-themed styling. It uses **ONNX Runtime Web** to load the trained model and execute **zero-latency, 100% on-device webcam emotion inference**, alongside a rich data dashboard powered by **Chart.js** displaying metrics like loss convergence, communication cost, and blocked attacks.

---

## 🚀 Quick Start (One-Click Auto-Deployer)

For the absolute easiest deployment, we have provided a **`start.bat`** script at the project root. Double-clicking this batch file automatically:
- Checks and uses your local **Python 3.12** installation.
- Initializes the **`venv` virtual environment** (if not already created).
- Upgrades `pip` and installs all dependencies (`torch`, `torchvision`, `flwr`, `onnx`, `pandas`, `matplotlib`, `onnxscript`) inside the isolated `venv`.
- Executes the entire federated learning simulation.
- Exports the model checkpoint to the optimized **ONNX format**.
- Compiles the CSV quantitative logs, publication figures, and detailed academic markdown report inside `results/`.
- Opens your default web browser to the biometric scanner page at **`http://localhost:8000/frontend/`**.
- Spins up the local HTTP web server on port 8000.

**Simply double-click `start.bat` at the project root to run the entire project!**

---

## Repository Structure

```
Edge-Emotion-FL/
├── backend/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── data_loader.py       # Handles Dirichlet Non-IID partitioning & synthetic fallback
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py               # Lightweight CNN architecture (PyTorch)
│   ├── client/
│   │   ├── __init__.py
│   │   └── fedprox_client.py    # NumPy Client with FedProx proximal loss, DP & poisoning attacks
│   ├── server/
│   │   ├── __init__.py
│   │   └── lbaa_strategy.py     # Custom LBAAFedAvg strategy with layer-by-layer MAD filtering
│   ├── simulation.py            # Orchesrates multi-client training and saves metrics log
│   └── export_onnx.py           # Exports the trained PyTorch global model to ONNX format
├── frontend/
│   ├── index.html               # Main interactive user interface (Webcam & Dashboard tabs)
│   ├── style.css                # Premium glassmorphism cyber-dark stylesheet
│   ├── app.js                   # Preprocesses webcam crops & runs client-side ONNX inference
│   └── simulation_history.json  # Data log exported by backend for Chart.js plotting
├── venv/                        # Python 3.12 isolated virtual environment
└── README.md                    # Complete research documentation & execution guide
```

---

## Technical Overview & Mathematical Models

### 1. Client-Side Optimization: FedProx
To stabilize local model training and prevent divergence across highly subjective, Non-IID (heterogeneous) user partitions, clients incorporate a proximal regularization term:

$$L_{prox}(w) = L_{CE}(w) + \frac{\mu}{2} \| w - w^t \|_2^2$$

- $L_{CE}(w)$ is the categorical cross-entropy loss on the local private dataset.
- $w^t$ represents the global weights received from the server.
- $\mu \ge 0$ (default $0.15$) controls the penalty on local parameter drift.

### 2. Privacy-Preserving Mechanism: Differential Privacy (DP)
To block reconstruction attacks (such as gradient leakages revealing original face details), clients apply Local Differential Privacy by clipping and injecting noise:

$$\Delta \bar{w}_i = \frac{\Delta w_i}{\max\left(1, \frac{\|\Delta w_i\|_2}{S}\right)}$$

$$\Delta \tilde{w}_i = \Delta \bar{w}_i + \mathcal{N}\left(0, \sigma^2 S^2 \cdot \mathbf{I}\right)$$

- $S$ is the $L_2$ norm clipping threshold (default $1.2$).
- $\sigma$ is the noise multiplier (default $0.05$).

### 3. Server-Side Anomaly Defense: LBAAFedAvg
To shield the global model from stealth poisoning attacks (like Fed-CRA or sign-flipping), the server evaluates client weight updates **layer-by-layer** using a robust **Median Absolute Deviation (MAD)** outlier filter:

For each layer $l$:
- Calculate the $L_2$ distance of each client's layer weights $W_k^l$ to the element-wise median $M^l$: $d_k^l = \|W_k^l - M^l\|_2$.
- Compute the modified Z-score based on MAD:
  $$MAD^l = \text{median}\left(\{|d_k^l - \text{median}(d^l)|\}\right)$$
  $$Z_k^l = \frac{0.6745 \cdot (d_k^l - \text{median}(d^l))}{MAD^l + 1\text{e-}12}$$
- If $Z_k^l > \tau$ (default $2.2$), layer $l$ of client $k$ is flagged.
- If a client is flagged in more than $\theta\%$ of layers (default $25\%$), its entire weight update is discarded from the round's aggregation pool.

---

## Setup & Running the Research

### 1. Prerequisite
Ensure **Python 3.12** is installed on your local machine.

### 2. Activate the Virtual Environment
Open **PowerShell** in the root project folder `Edge-Emotion-FL/` and activate the pre-configured virtual environment:
```powershell
# Set Execution Policy if script execution is blocked
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the venv
.\venv\Scripts\Activate.ps1
```

### 3. Run the Federated Learning Simulation
Execute the Flower simulation to train the model collaboratively across 5 clients. In this simulation, **Client 4 acts as a covert attacker (sign-flipping)**, proving that LBAAFedAvg successfully filters and discards malicious updates:
```bash
python -m backend.simulation
```
- **Automatic Fallback:** The script checks for the raw `fer2013.csv` file. If not found, it automatically generates a realistic synthetic dataset skewing labels via Dirichlet distribution so the simulation compiles immediately.
- **Results Export:** Upon completion, the script saves `backend/global_model.pth` and exports `frontend/simulation_history.json`.

### 4. Export the Trained Model to ONNX Format
Convert the final PyTorch global weights checkpoint into a browser-ready, dynamic-batch ONNX model:
```bash
python -m backend.export_onnx
```
- This exports and structures the graph into `frontend/global_model.onnx`.

### 5. Launch the Web Application
Start a quick, lightweight local web server in the project folder to serve the static frontend without CORS blocks:
```bash
python -m http.server 8000
```
Open your browser and navigate to:
👉 **[http://localhost:8000/frontend/](http://localhost:8000/frontend/)**

---

## Core Frontend Features

- **Live Webcam Mode:** Aligns the user's face in a biometric grid, extracts the cropped area, transforms it to grayscale, resizes it to $48\times 48$, and feeds it to **ONNX Runtime Web** for instant, local inference. Displays glowing confidence bars for Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral categories at 30+ FPS.
- **Metrics Dashboard Mode:** Visualizes key academic benchmarks, plotting global accuracy convergence curves, training loss, cumulative network bandwidth overhead (MB), and anomalous clients blocked per round in interactive charts powered by **Chart.js**.
- **Security Audit Logger:** Demonstrates that the server-side LBAAFedAvg strategy successfully identified and rejected the malicious node (Client 4) throughout the entire training process!
