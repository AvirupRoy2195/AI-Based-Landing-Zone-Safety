# 🚁 AI-Based Landing Zone Safety

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered system for autonomous drone landing zone classification using terrain features from aerial imagery.

![Safety Heatmap](https://via.placeholder.com/800x400?text=Landing+Safety+Heatmap)

## 📋 Overview

This project implements a machine learning pipeline to classify drone landing zones as **Safe** or **Unsafe** based on terrain characteristics extracted from aerial imagery. The system prioritizes **Precision** over Accuracy to minimize dangerous False Positives in safety-critical autonomous drone operations.

### Key Features
- 📊 **Comprehensive EDA & Interpretation**: Detailed statistical analysis with domain-specific risk assessments
- 📝 **Detailed Notebook Documentation**: Step-by-step markdown interpretations linking data to safety conclusions
- 🎯 **High-Accuracy Classification**: Ensemble models achieving ~97% ROC-AUC
- 🔧 **Advanced Feature Engineering**: Domain-specific interaction features
- 📊 **Spatial Safety Mapping**: Heatmap visualization of the "Goldilocks Zone"
- 🧠 **Explainable AI**: SHAP analysis for transparent decision-making
- 🚀 **Autonomous Decision Logic**: Ready-to-deploy fallback behaviors

## 🗂️ Project Structure

```
AI-Based-Landing-Zone-Safety/
├── Capstone_Project.ipynb      # Main analysis notebook (all code + outputs)
├── Landing Zone Dataset.xlsx   # Training dataset
├── AI Based Landing Zone Safety Capstone.pdf  # Project documentation
├── README.md                   # This file
├── ARCHITECTURE.md             # Technical architecture details
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/AvirupRoy2195/AI-Based-Landing-Zone-Safety.git
cd AI-Based-Landing-Zone-Safety

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Capstone_Project.ipynb
```

## 📊 Dataset Features

| Feature | Description | Safety Impact |
|---------|-------------|---------------|
| `slope_deg` | Terrain slope (degrees) | High slope → Toppling risk |
| `roughness` | Surface texture variance | High roughness → Landing gear damage |
| `edge_density` | Visual edge frequency | Complex terrain = obstacles |
| `ndvi_mean` | Vegetation index | High NDVI = Tall vegetation |
| `shadow_fraction` | Shadow coverage | Shadows mask obstacles |
| `object_density` | Detected objects count | Objects = Obstacles |
| `confidence_score` | Detection confidence | Low confidence = uncertain |

## 🔧 Methodology

### 1. Feature Engineering
Created 4 interaction features to capture complex terrain patterns:
- `slope_roughness`: Combined terrain difficulty
- `terrain_complexity`: Visual complexity score
- `safety_index`: Confidence-adjusted safety indicator
- `vegetation_shadow`: Hidden vegetation risk

### 2. Model Optimization
- **SMOTE**: Synthetic oversampling for class balance
- **Model Comparison**: RF, XGBoost, GradientBoosting, LightGBM
- **Stacking Ensemble**: RF + XGBoost + GB with LogisticRegression meta-learner
- **Voting Ensemble**: Final model combining best performers

### 3. Evaluation Metrics

| Metric | Score | Significance |
|--------|-------|--------------|
| Accuracy | ~93% | Overall correctness |
| Precision | ~93% | Low False Positives (critical for safety) |
| ROC-AUC | ~97% | Excellent discrimination |

## 🗺️ Spatial Safety Analysis

The system generates a **Goldilocks Zone** heatmap showing optimal landing conditions:

- **🟢 Green Zone**: Slope < 10°, Roughness < 0.2 → Safe for landing
- **🟡 Yellow Zone**: Moderate risk → Proceed with caution
- **🔴 Red Zone**: High slope + roughness → Landing prohibited

## 🤖 Autonomous Decision Logic

```
IF Safety >= 0.85 AND Confidence >= 0.70: → LAND
ELIF Safety >= 0.60:                      → LAND_CAUTION
ELIF Safety >= 0.40:                      → LOITER & REASSESS
ELSE:                                     → ABORT & SEEK ALTERNATIVE
```

### Fallback Behaviors
| Scenario | Action |
|----------|--------|
| Low confidence | Acquire more frames |
| High Roughness | **Spiral Search** for neighbors |
| High slope detected | Search flatter terrain |
| No safe zone found | Return to launch (RTL) |

## 📈 Results

The optimized pipeline achieves:
- **Minimized False Positives**: Critical for preventing drone crashes
- **Well-Calibrated Probabilities**: Reliable for threshold-based decisions
- **Interpretable Predictions**: SHAP explains feature contributions

## 🔮 Future Improvements

| Limitation | Proposed Solution |
|------------|-------------------|
| Static features | Real-time video processing |
| Optical only | LiDAR/Radar integration |
| Single viewpoint | Multi-view imagery |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Avirup Roy**
- GitHub: [@AvirupRoy2195](https://github.com/AvirupRoy2195)

## 🙏 Acknowledgments

- Capstone project for AI-based safety systems
- XGBoost and scikit-learn communities
- SHAP library for model interpretability
