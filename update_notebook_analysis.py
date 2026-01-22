
import json
import os

notebook_path = r"d:\VSCODE\Capstone_AI Based Landing Zone Safety\Capstone_Project.ipynb"

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new contents
eda_text = [
    "### ✅ Descriptive Analytics Interpretation & Operational Insights\n",
    "\n",
    "**1. Statistical Risk Profile**\n",
    "*   **Terrain Slope**: The mean slope is ~8.3°, which is well within standard commercial drone limits (typically <15°). However, the max of 27.6° indicates the presence of \"No-Go\" zones that must be strictly filtered.\n",
    "*   **Roughness & Debris**: Roughness values > 0.4 generally correlate with rocky or vegetation-heavy terrain, posing a tipping hazard during touchdown.\n",
    "*   **Confidence**: An average score of 0.77 suggests the perception system is effective, but the tail of low-confidence scores (<0.6) requires a \"Loiter & Reassess\" fallback strategy.\n",
    "\n",
    "**2. Distribution Analysis (Histograms)**\n",
    "*   **Safety Distribution**: The clear separation in Slope and Roughness histograms validates them as primary discriminators.\n",
    "    *   **Safe Zones (Green)**: Tightly clustered near 0 slope/roughness.\n",
    "    *   **Unsafe Zones (Red)**: Long tail distributions. This non-linearity suggests that linear models might struggle, justifying our use of Tree-based ensembles (Random Forest/XGBoost).\n",
    "\n",
    "**3. Outlier Warnings (Box Plots)**\n",
    "*   **Critical Thresholds**: The box plots suggest a \"Safety Cutoff\" around 12° slope and 0.35 roughness. Data points above these values are overwhelmingly labeled 'Unsafe'.\n",
    "*   **Sensor Noise**: outliers in `confidence_score` for the 'Safe' class (low confidence despite safe terrain) imply potential sensor artifacts (e.g., lighting glare), reinforcing the need for the `safety_index` feature to weight predictions.\n"
]

feat_eng_text = [
    "### ✅ Feature Engineering: Capturing Compound Risks\n",
    "**Why create interaction features?**\n",
    "*   **`slope_roughness` (Slope × Roughness)**: *Physics-based Rationale*: A steep slope is risky, and a rough surface is risky. But a steep, rough surface is **exponentially** more dangerous (high tipping risk + low traction). This interaction feature helps the model identifying these compound \"Kill Zones\" that linear features might miss.\n",
    "*   **`terrain_complexity` (Edge Density × Object Density)**: Proxies for \"Visual Clutter\". High values likely indicate dense vegetation or rubble fields where landing sensors might be confused.\n",
    "*   **`safety_index` (Confidence / (Slope + 1))**: A \"Reliability Metric\". It penalizes high-confidence scores if the terrain is steep, ensuring we don't blindly trust the perception system in geometric edge cases.\n",
    "*   **`vegetation_shadow` (NDVI × Shadow)**: Highlights areas where vegetation might be hiding obstacles in shadows—a common cause of \"Phantom Safe\" classifications."
]

smote_text = [
    "### ✅ Sampling Strategy Interpretation\n",
    "*   **Pre-SMOTE Status**: The dataset was nearly balanced (1350 vs 1650), but slight imbalances can still bias precision-critical models.\n",
    "*   **Why SMOTE?**: Synthetic Minority Over-sampling Technique (SMOTE) creates synthetic examples of the minority class in feature space. This \"fills in the gaps\" between real examples, forcing the model to learn broader, more generalized decision boundaries rather than overfitting to specific minority instances.\n",
    "*   **Goal**: Ensure the model is as sensitive to \"Safe\" spots as it is to \"Unsafe\" ones, maximizing our ability to find landing zones in difficult terrain."
]

corr_cell_source = [
    "### ✅ Feature Correlation Interpretation\n",
    "**Key Drivers of Safety:**\n",
    "*   **Negative Correlations (Risk Factors)**: `slope_roughness` (-0.363) and `slope_deg` (-0.312) are the strongest predictors of an **Unsafe** label. This confirms that topological relief is the dominant safety constraint.\n",
    "*   **Positive Correlations (Safety Indicators)**: `safety_index` (+0.208) is the strongest positive correlate, validating our engineered feature's utility. `ndvi_mean` also correlates with safety, likely because open grassy fields (high NDVI) are safer than rocky outcrops (low NDVI).\n",
    "*   **Insight**: The strong correlation of engineered features (specifically the interaction terms) justifies their inclusion and suggests they will play a significant role in Tree/Forest feature importance.\n"
]

new_corr_cell = {
   "cell_type": "markdown",
   "id": "corr_interp",
   "metadata": {},
   "source": corr_cell_source
}

# Apply updates
cells = nb['cells']
new_cells = []
correlation_inserted = False
updates_made = 0

for cell in cells:
    # Update EDA
    if cell.get('id') == 'ed7837e9':
        cell['source'] = eda_text
        updates_made += 1
        print("Updated EDA cell.")
    
    # Update Feature Engineering
    if cell.get('id') == 'f269f68c':
        cell['source'] = feat_eng_text
        updates_made += 1
        print("Updated Feature Engineering cell.")
        
    # Update SMOTE
    if cell.get('id') == '4d9e5e8f':
        cell['source'] = smote_text
        updates_made += 1
        print("Updated SMOTE cell.")
        
    new_cells.append(cell)
    
    # Insert Correlation cell after code cell '4d8defd5'
    if cell.get('id') == '4d8defd5' and not correlation_inserted:
        new_cells.append(new_corr_cell)
        correlation_inserted = True
        updates_made += 1
        print("Inserted Correlation cell.")

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook updated successfully with {updates_made} changes.")
