# HCI-Tasks

This repository contains the implementation, signal processing, and analysis for various Human-Computer Interaction (HCI) tasks. The projects focus on bridging the gap between physiological signals (EOG, ECG) and software systems through feature extraction and GUI-based visualization.

## 📑 Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)
* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [Requirements](https://www.google.com/search?q=%23requirements)
* [Task Documentation](https://www.google.com/search?q=%23task-documentation)

---

## 💡 Project Overview

These tasks were developed to explore the extraction and classification of biological signals. Key focus areas include:

* **EOG (Electrooculography):** Mapping ocular signals to interface commands.
* **ECG (Electrocardiography):** Processing heart-rate data for human-state recognition.
* **Feature Engineering:** Transforming raw sensor data into meaningful feature matrices for machine learning models.

---

## 📂 Project Structure

```text
HCI-Tasks/
├── Task 1_EOG Task/        # EOG signal processing & GUI integration
│   ├── GUI.py              # Interactive visualization dashboard
│   ├── Task1.py            # Filtering, feature extraction, and analysis
│   └── features_matrix.xlsx# Processed feature dataset
└── Task 2_ECG Task/        # ECG signal analysis
    ├── ECG.py              # Heart rate variability and noise reduction logic
    └── ECG_Features.xlsx   # Compiled heartbeat feature sets

```

---

## ⚙️ Requirements

To run these scripts, you will need **Python 3.x** and the following dependencies:

```bash
# Install required libraries
pip install pandas numpy matplotlib scipy

```

---

## 📝 Task Documentation

### Task 1: EOG Signal Classifier

* **Objective:** Extract horizontal eye movement patterns from raw signal text files.
* **Implementation:** The pipeline includes signal denoising followed by a peak-detection algorithm. The `GUI.py` allows for the inspection of raw versus processed signal states.

### Task 2: ECG Analysis

* **Objective:** Perform feature extraction on heart rate data.
* **Implementation:** `ECG.py` utilizes signal filtering techniques to remove artifacts, enabling precise R-peak detection and feature calculation.
