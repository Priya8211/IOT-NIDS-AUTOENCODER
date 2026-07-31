# 📡 IoT Network Anomaly Detection Dashboard (N-BaIoT)

A real-time, AI-powered Network Intrusion Detection System (NIDS) designed to monitor IoT devices for cyber attacks (such as Mirai and Gafgyt botnets). 

This project combines the **Kitsune feature extraction algorithm** with a **Deep Learning Autoencoder** to detect network anomalies in real-time. Instead of relying on static rules, it uses an **Adaptive Thresholding** mechanism to dynamically adjust to normal network fluctuations, significantly reducing false positives.

## ✨ Key Features
- 🧠 **Deep Learning Autoencoder:** Trained exclusively on benign (normal) IoT traffic to reconstruct data. High reconstruction errors indicate malicious activity.
- 📡 **Live Packet Sniffing:** Uses `Scapy` in a background thread to capture and analyze raw network packets in real-time.
- 📊 **Kitsune Feature Extraction:** Extracts 115 statistical features on-the-fly using Damped Windows (1D/2D) to track packet sizes, jitter, and connection frequencies.
- 📈 **Adaptive Thresholding:** Dynamically calculates the anomaly threshold using a rolling mean and standard deviation, adapting to "concept drift" in network traffic.
- 🖥️ **Premium Cyber Dashboard:** A beautiful, dark-themed, real-time Streamlit UI featuring live charts, severity breakdowns, system health gauges, and an exportable alert log.
- 🔄 **Training Mode**: Train custom autoencoders on specific IoT device datasets
- 📥 **Export Alerts**: Download detected anomalies as CSV reports

## 🛠️ Tech Stack
- **Core AI:** TensorFlow / Keras, Scikit-Learn
- **Network Analysis:** Scapy
- **Frontend / UI:** Streamlit, Plotly, Custom CSS
- **Data Processing:** Pandas, NumPy
- **Feature Extraction**: Kitsune Algorithm (Damped Windows)
  
## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nbaiot-autoencoder.git
cd nbaiot-autoencoder
```

### 2. Create virtual enviornment

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
nbaiot-autoencoder/
├── Danmini_Doorbell/
│   ├── benign_traffic.csv
│   ├── gafgyt_attacks/
│   ── mirai_attacks/
├── Ecobee_Thermostat/
└── ... (other devices)

