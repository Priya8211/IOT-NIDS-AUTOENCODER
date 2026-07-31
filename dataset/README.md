   # Dataset Information
   This project uses the **N-BaIoT Dataset**. 
   Due to GitHub's size limits, the full dataset is not included here.
   
   📥 **Download the full dataset from here: ** [[https://drive.google.com/file/d/1aoU37iTMvSQCq1YWtCUVfq6UcK-ntzFl/view?usp=sharing]]
   
   Extract the contents into this `dataset/` folder so the structure looks like:
   dataset/
   ├── Danmini_Doorbell/
   │ ├── benign_traffic.csv
   │ ├── gafgyt_attacks/
   │ │ ├── gafgyt_combo.csv
   │ │ └── ...
   │ ── mirai_attacks/
   │ ├── mirai_ack.csv
   │ └── ...
   ├── Ecobee_Thermostat/
   ├── Ennio_Doorbell/
   ├── Philips_B120N10_Baby_Monitor/
   ├── Provision_PT_737E_Security_Camera/
   ├── Provision_PT_838_Security_Camera/
   ├── Samsung_SNH_1011_N_Webcam/
   ├── SimpleHome_XCS7_1002_WHT_Security_Camera/
   └── SimpleHome_XCS7_1003_WHT_Security_Camera/

   ##  Dataset Info
   
   - **Total Devices:** 9 IoT devices
   - **Features:** 115 statistical features extracted by Kitsune algorithm
   - **Attacks Included:** 
     - **Mirai** (ACK, Scan, UDP, UDPplain, HTTP)
     - **Gafgyt** (Combo, Junk, Scan, TCP, UDP)
   - **File Format:** CSV (Comma-Separated Values)
   
   ## 🔧 Quick Start
   
   1. Download the dataset
   2. Place all device folders in this directory
   3. Run the Streamlit app: `streamlit run app.py`
   4. Select your device from the sidebar dropdown
   5. Click "▶️ Start Live Stream" to begin monitoring
   
   ## 📄 File Descriptions
   
   - **benign_traffic.csv**: Normal, non-malicious network traffic (used for training)
   - **gafgyt_attacks/**: Folder containing Gafgyt botnet attack traffic
   - **mirai_attacks/**: Folder containing Mirai botnet attack traffic
