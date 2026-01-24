# AI-Based Landing Zone Safety Capstone

## Project Overview
This project develops an autonomous AI system to identify safe landing zones for UAVs. It utilizes machine learning to analyze terrain features (slope, roughness, vegetation) and predict landing safety with a specific focus on **minimizing fatal errors**.

## Key Files
*   **`Capstone_Project.ipynb`**: The main Jupyter Notebook. It contains the end-to-end pipeline:
    *   Data Loading & EDA
    *   Feature Engineering & Selection
    *   Multi-Model Training & Safety Evaluation
    *   **Zero-Fatal-Error Ensemble** Construction
    *   Spatial Visualization (Heatmaps) & Autonomy Logic
*   **`Landing Zone Dataset.xlsx`**: The source dataset containing 3000 terrain samples.
*   **`AI Based Landing Zone Safety Capstone.pdf`**: Original project requirements.

## How to Run
1.  Open `Capstone_Project.ipynb` in VS Code or Jupyter Lab.
2.  Ensure dependencies are installed (`pandas`, `sklearn`, `seaborn`, `xgboost`, `imblearn`).
3.  Run all cells. The notebook will automatically:
    *   Train 9+ models.
    *   Identify the safest models (0 False Positives).
    *   Build a Voting Ensemble.
    *   Generate all charts and decision metrics.

## Methodology: "The Judicious Move"
The modeling strategy prioritizes **Safety First**. Models are ranked primarily by their number of False Positives (Fatal Errors). The final ensemble is constructed exclusively from models that demonstrated **Zero Fatal Errors** during testing, ensuring the drone never attempts a landing in a truly unsafe zone.
