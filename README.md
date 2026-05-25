# Robot Sensor Anomaly Detector

Detects faults in robot sensor streams using an LSTM Autoencoder.
Trained on simulated temperature, vibration, and voltage time-series
with 4 injected fault types — outperforms Isolation Forest baseline
by 10 F1 points (F1=0.82, AUC=0.89).

**Live demo:** http://your-ec2-domain.com:8000/docs

## Architecture
[paste your architecture diagram here]

## Results
| Model            | F1    | ROC-AUC |
|------------------|-------|---------|
| LSTM Autoencoder | 0.818 | 0.887   |
| Isolation Forest | 0.718 | 0.813   |

## Stack
Python · PyTorch · FastAPI · React · Firebase · Docker · AWS EC2

## Quick start
\`\`\`bash
git clone https://github.com/<you>/robot-sensor-anomaly-detector
cd robot-sensor-anomaly-detector
pip install -r requirements.txt
python sensor_simulator.py    # generate data
python anomaly_model.py       # train model
uvicorn main:app --reload     # start API → localhost:8000/docs
\`\`\`

## Project structure
[brief description of each file]