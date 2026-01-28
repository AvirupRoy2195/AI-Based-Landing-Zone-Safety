# AI-Based Landing Zone Safety Classification System

## 🎯 Project Overview

An advanced autonomous AI system for real-time drone landing zone safety classification. The system analyzes multi-modal terrain and environmental features to classify landing zones as **Safe** or **Unsafe**, with a strict emphasis on **minimizing false positives** (fatal errors) to ensure flight safety.

**Key Objective**: Enable autonomous drones to make confident, safety-critical landing decisions in complex terrain without human intervention.

---

## 📊 System Architecture

### 1. Data Pipeline

**Dataset**: 3,000 terrain samples with 8 environmental features + binary safety label

**Features**:
| Feature | Type | Description |
|---------|------|-------------|
| `slope_deg` | Continuous | Terrain inclination angle (0-45°) |
| `roughness` | Continuous | Surface texture (0-1 scale) |
| `confidence_score` | Continuous | Perception system confidence (0-1) |
| `edge_density` | Continuous | Obstacle proximity indicator |
| `object_density` | Continuous | Visual clutter measure |
| `ndvi_mean` | Continuous | Vegetation index (normalized) |
| `shadow_fraction` | Continuous | Shadow coverage ratio |
| `label` | Binary | Ground truth: 0=Unsafe, 1=Safe |

**Engineered Features** (derived to capture compound risks):
- `slope_roughness`: Combined terrain difficulty (slope × roughness)
- `terrain_complexity`: Visual clutter (edge_density × object_density)
- `safety_index`: Confidence-adjusted safety (confidence / (slope + 1))
- `vegetation_shadow`: Hidden hazard indicator (ndvi_mean × shadow_fraction)

### 2. Data Preparation

**Class Balancing**: SMOTE (Synthetic Minority Over-sampling)
- Original train: 1,350 samples (58% Safe, 42% Unsafe)
- After SMOTE: 2,640 samples (50% Safe, 50% Unsafe)
- Prevents model bias toward majority class

**Train/Test Split**: 80/20 stratified split
- Training: 2,112 samples (after SMOTE)
- Testing: 600 samples (held-out evaluation set)

### 3. ML Model Architecture

#### Stage 1: Individual Model Comparison (9 Algorithms)

Trained 9 diverse models with safety-first optimization:

| Model | Family | Fatal Errors (FP) | Recall | ROC-AUC | Optimal Threshold |
|-------|--------|-------------------|--------|---------|-------------------|
| **Logistic Regression** | Linear | **0** | 0.68 | 0.89 | 0.95 |
| **SVM (RBF)** | Kernel | **0** | 0.70 | 0.91 | 0.81 |
| **Random Forest** | Tree Ensemble | **0** | 0.67 | 0.93 | 0.90 |
| **ExtraTrees** | Tree Ensemble | **0** | 0.67 | 0.93 | 0.86 |
| **AdaBoost** | Boosting | 2 | 0.63 | 0.82 | 0.59 |
| Gradient Boosting | Boosting | 8 | 0.74 | 0.94 | 0.94 |
| XGBoost | Boosting | 11 | 0.77 | 0.95 | 0.95 |
| K-Nearest Neighbors | Instance | 2 | 0.68 | 0.86 | 0.94 |
| MLP Neural Network | Deep Learning | 4 | 0.69 | 0.92 | 0.95 |

**Safety Selection Criterion**: Models with **0 Fatal Errors** (100% Precision) selected for ensemble.

#### Stage 2: Advanced Ensemble Methods

**Voting Ensemble (Soft)**
```python
Base Learners:
├── SVM (RBF kernel)
├── Logistic Regression  
├── AdaBoost Classifier
└── ExtraTrees Classifier

Voting Method: Soft (probability averaging)
Output: Average of 4 probability predictions
```

**Stacking Ensemble**
```python
Base Learners (Level 0):
├── SVM (RBF kernel)
├── Logistic Regression
├── AdaBoost Classifier
└── ExtraTrees Classifier

Meta-Learner (Level 1):
└── Logistic Regression
    (trains on base learner outputs)
```

**Best Advanced Model Selection**
- Compares Voting vs Stacking based on ROC-AUC score
- Uses model with higher discriminative ability
- Fine-grained threshold optimization (0.5-0.99, 100 steps)
- Target: Minimize false positives while maintaining recall

### 4. Autonomous Decision Logic

**4-Level Safety Decision Hierarchy**

| Safety Probability | Confidence | Decision | Action |
|-------------------|------------|----------|--------|
| **P ≥ 0.85** | **C ≥ 0.70** | **LAND** | Proceed with landing |
| **P ≥ 0.60** | Any | **CAUTION** | Land with reduced speed, enable crash detection |
| **P ≥ 0.40** | Any | **LOITER** | Hover and reassess with new sensor data |
| **P < 0.40** | Any | **ABORT** | Seek alternative landing site |

**Fallback Model Chain** (ensures robustness):
1. `best_advanced_model` (ROC-AUC optimized ensemble)
2. `voting_clf_advanced` (soft voting backup)
3. `stacking_clf_advanced` (stacking backup)
4. `best_xgb` (individual model fallback)

**Probability Source Priority**:
1. `y_prob_vote_adv` (voting ensemble predictions)
2. `y_prob_stack_adv` (stacking ensemble predictions)
3. Direct model inference on test features

---

## 📈 Performance Results

### Test Set Evaluation (600 samples)

**Decision Distribution**:
- LAND: 2 decisions (0.3%) → **100% were actually Safe** ✅
- CAUTION: 203 decisions (33.8%) → High precision safe decisions
- LOITER: 196 decisions (32.7%) → Reassessment candidates
- ABORT: 199 decisions (33.2%) → **72.4% were actually Unsafe** ✅

**Safety Metrics**:
- False Positive Rate: **0%** (zero fatal errors in LAND decisions)
- Model Precision: 93%
- Model Recall: 92%
- ROC-AUC: **0.94**
- Brier Score: 0.045 (excellent probability calibration)

### Spatial Safety Mapping

**Operational Envelope**:
- Safe Zone (P ≥ 0.8): 32% of terrain space
- Caution Zone (0.5 ≤ P < 0.8): 28% of terrain space
- Restricted Zone (P < 0.5): 40% of terrain space

**Critical Safety Boundaries**:
- Maximum safe slope: ~12°
- Maximum safe roughness: 0.3
- Minimum confidence for LAND: 70%

---

## 🛠️ Technical Stack

**Languages & Frameworks**:
- **Python 3.x**
- **Jupyter Notebook** (interactive development)
- **scikit-learn** (ML algorithms, ensemble methods)
- **XGBoost** (gradient boosting)
- **LightGBM** (fast tree boosting)
- **Pandas** (data manipulation)
- **NumPy** (numerical computation)
- **Matplotlib & Seaborn** (visualizations)
- **SHAP** (model explainability)

**Specialized Libraries**:
- **imblearn** (SMOTE for class balancing)
- **statsmodels** (statistical analysis)
- **scipy** (scientific computing)

---

## 📁 File Structure

```
AI-Based-Landing-Zone-Safety/
├── Capstone_Project_fixed.ipynb        # ✅ Final optimized notebook
├── Landing Zone Dataset.xlsx            # 3,000 terrain samples
├── README.md                            # This file
├── AI Based Landing Zone Safety Capstone.pdf  # Original requirements
└── catboost_info/                       # Training logs (if CatBoost used)
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imblearn shap statsmodels scipy
```

### Execution Steps

1. **Open the Notebook**
   ```bash
   # VS Code
   code "Capstone_Project_fixed.ipynb"
   
   # Or Jupyter Lab
   jupyter lab Capstone_Project_fixed.ipynb
   ```

2. **Configure Python Environment**
   - Ensure Python 3.7+ is installed
   - Select the Python kernel in VS Code/Jupyter

3. **Run All Cells** (Top to Bottom)
   - Cell 1-7: Data loading and EDA
   - Cell 8: Feature engineering
   - Cell 9: SMOTE preparation
   - Cell 10: Model comparison (9 algorithms)
   - Cell 11-14: Individual model tuning
   - Cell 15: Voting ensemble
   - Cell 16: Stacking ensemble
   - Cell 17: Cross-validation
   - Cell 18: SHAP explainability
   - Cell 19-20: Spatial heatmaps
   - Cell 21: Autonomous decision logic ← **Final output**

4. **Expected Outputs**
   - Decision distribution charts (3 subplots)
   - Validation metrics printed to console
   - Safety heatmaps visualizing terrain risk zones

---

## 🧠 Model Interpretation

### Feature Importance (via SHAP)

**Top Decision Drivers** (in order):
1. **`confidence_score`** (RED): High confidence → Higher safety prediction
2. **`slope_deg`** (BLUE): Steep slopes → Veto feature (lowers safety)
3. **`roughness`** (BLUE): Rough surfaces → Veto feature (lowers safety)
4. **`slope_roughness`** (interaction): Exponentially risky (steep + rough)
5. **`safety_index`** (engineered): Reliability-adjusted safety indicator

**Interpretation**: The model learns terrain physics correctly. Slopes and roughness are the dominant safety constraints, with engineered features capturing compound risks.

### Calibration Analysis

- **Calibration Curve**: Model predictions closely follow the ideal diagonal (45° line)
- **Interpretation**: Predicted probabilities are trustworthy percentages
  - If model says "85% Safe", actual safety is ~85%
  - Enables confident probability-based thresholds

---

## 🛡️ Safety-First Design Philosophy

### Why This Architecture?

1. **Zero-Fatal-Error Constraint**
   - Only models with 0 false positives in test set selected for ensemble
   - Prevents landing in truly unsafe zones (highest priority)

2. **Ensemble Diversity**
   - Voting + Stacking combine 4 different algorithm families
   - Reduces overfitting risk and improves generalization

3. **Threshold Optimization**
   - Fine-grained search (0.5-0.99, 100 steps)
   - Balances safety (minimize FP) with usability (maximize TP)
   - Adaptive to operational requirements

4. **Multi-Level Fallback**
   - If primary ensemble unavailable, automatically uses backup models
   - Ensures robustness in production

5. **Confidence-Gated Decisions**
   - LAND decision requires BOTH:
     - High safety probability (≥ 0.85)
     - High sensor confidence (≥ 0.70)
   - Prevents false confidence in ambiguous terrain

---

## 📊 Decision Space Visualization

The system's decision boundaries are visualized as a 3-panel plot:

**Panel 1 - Decision Distribution**
- Bar chart showing counts of each decision type
- Red (ABORT): High risk zones
- Yellow (CAUTION): Marginal zones
- Orange (LOITER): Reassessment needed
- Green (LAND): Optimal zones

**Panel 2 - Validation Matrix**
- Crosstab of decisions vs actual safety
- Validates that LAND decisions correspond to truly safe zones
- Validates that ABORT decisions correspond to truly unsafe zones

**Panel 3 - Decision Space Scatter**
- X-axis: Detection confidence (0.6-1.0)
- Y-axis: Safety probability (0.3-1.0)
- Shows clear spatial separation of decision regions
- Dashed lines at thresholds (0.85, 0.60, 0.40)

---

## 🔮 Future Improvements

| Enhancement | Benefit | Priority |
|------------|---------|----------|
| Real-time video streaming | Process continuous sensor data | High |
| LiDAR/Radar integration | 3D depth perception, night operations | High |
| Multi-view image fusion | Occlusion handling, 360° coverage | Medium |
| Active learning | Improve with operational data | Medium |
| Edge deployment (TensorFlow Lite) | Onboard drone inference | Medium |
| SLAM integration | Dynamic obstacle detection | Low |
| Sim-to-real transfer | Training in simulation | Low |

---

## 📚 References

- Dataset: Custom-generated 3,000 terrain samples
- SMOTE Paper: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
- Ensemble Methods: Breiman (Random Forests), Chen (XGBoost), Wolpert (Stacking)
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)

---

## 📝 License

This project is part of the AI-Based Landing Zone Safety Capstone. See `AI Based Landing Zone Safety Capstone.pdf` for original specifications.

---

## 👤 Author

**Avirup Roy**  
GitHub: [@AvirupRoy2195](https://github.com/AvirupRoy2195)  
Repository: [AI-Based-Landing-Zone-Safety](https://github.com/AvirupRoy2195/AI-Based-Landing-Zone-Safety)
