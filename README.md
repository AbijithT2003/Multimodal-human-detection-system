
# Multimodal Human Detection System using Deep Learning Methods 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange?logo=pytorch)
![Transformer](https://img.shields.io/badge/Model-DETR-brightgreen?logo=transformer)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🔍 Overview

This project presents a **Transformer-based human detection system** that integrates **RGB, Thermal, and Infrared (IR) imaging** for robust detection in diverse environmental conditions. Built using the DEtection TRansformer (DETR) model, the system improves detection accuracy in scenarios such as low-light, occlusion, and cluttered backgrounds.

### 👨‍💻 Developed By
- **Abijith T**
- Agney Suresh  
- Athulkrishna Dhananjayan  
*(Under the guidance of Dr. Sreelekshmi P S, Assistant Professor, ECE Dept., NSS College of Engineering)*

## 🧠 Key Features

- 🔁 **Multimodal Input:** RGB, Thermal, and Infrared imaging
- ⚙️ **Advanced Preprocessing:** CLAHE, Histogram Equalization, Gaussian Filtering
- 🧭 **Object Detection with DETR:** End-to-end transformer model with self-attention
- 🔧 **Hyperparameter Tuning:** Implemented using Optuna framework
- 📊 **Model Evaluation:** mAP, IoU, GIoU, training/validation curves
- 🆚 **Comparative Analysis:** DETR vs CNN-based models (YOLO, Faster R-CNN)

## 📁 Project Structure

```
├── dataset/
│   ├── RGB/
│   ├── Thermal/
│   ├── IR/
├── src/
│   ├── preprocessing.py
│   ├── detr_model.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── loss_curves/
│   ├── detections/
│   └── metrics/
├── notebooks/
│   ├── Data Exploration.ipynb
│   └── Optuna_Tuning.ipynb
├── README.md
└── requirements.txt
```

## 📷 Sample Visuals

| Raw Image | Preprocessed | Detection Output |
|-----------|--------------|------------------|
| ![rgb](results/sample_rgb.png) | ![clahe](results/sample_clahe.png) | ![detection](results/sample_detection.png) |

## 🧰 Tech Stack

- **Language:** Python 3.10
- **Frameworks/Libraries:** PyTorch, OpenCV, Matplotlib, Optuna
- **Model Architecture:** Detection Transformer (DETR)
- **Annotation Format:** COCO
- **Hardware:** Raspberry Pi + Thermal/IR Cameras

## 🧪 Performance Metrics

| Modality        | IoU (%) | mAP (%) |
|----------------|---------|---------|
| RGB Only       | 58.3    | 61.2    |
| Thermal Only   | 60.1    | 63.0    |
| IR Only        | 57.4    | 60.0    |
| **Multimodal** | **72.5** | **76.8** |

## 📦 Installation

```bash
git clone https://github.com/abijitht/multimodal-human-detection
cd multimodal-human-detection
pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Training the model
python src/train.py --config configs/multimodal.yaml

# Evaluating the model
python src/evaluate.py --weights saved_models/detr_multimodal.pth
```

## 📚 References

- [DETR: End-to-End Object Detection with Transformers (Facebook AI)](https://arxiv.org/abs/2005.12872)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [OpenThermalPose Dataset](https://github.com/Geng-J/OpenThermalPose)

## 🔗 Connect with Me

📫 [abijitht.dev@gmail.com](mailto:abijitht.dev@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/abijitht/)  
📸 [Instagram](https://instagram.com/abijith_t)  
🌐 [Portfolio (Coming Soon)]()

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
