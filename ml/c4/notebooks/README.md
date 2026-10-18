# C4 research notebooks

Google Colab notebooks of Component 4 (Sandanayaka S D I K – IT23222472), with the outputs of their
final Colab runs. Data and model files stay on Google Drive and are never committed.

| Notebook | Purpose |
|---|---|
| 01_Data_Acquisition_and_Synchronization | Load the 9 PVS trips, check timestamps/sampling, synchronise left/right sensors, worst-side label |
| 02_EDA_and_Data_Quality | Data quality, class balance, noise sources, speed/orientation evidence (Section 10: decisions) |
| 03_Signal_Preprocessing_and_Noise_Handling | Glitch handling, gravity removal, 40 Hz low-pass, invariant channels, windows |
| 04_Feature_Engineering | Time and spectral features, stationary noise-profile subtraction |
| 05_Normalization_and_Data_Preparation | Speed normalisation, per-trip normalisation, stable feature selection |
| 06_ML_Model_Training_and_Generalization | SVM / RF / XGB; Random, LOVO, LORO, LOVO+LORO; ablation C0–C7 |
| 07_Generalization_Analysis | Final C6/C7 results, placement, sensitivity, speed ablation, deployment model |
| 08_Feature_Importance_Analysis | Permutation importance and fold stability of the selected features |
| 09_GPS_Based_Road_Roughness_Visualization | Map of predictions vs labels |
| 10_Export_Deployment_Model | Deployment model (all trips) → portable format for the app (`ml/c4/models/production`) |
| 11_Unseen_Vehicle_App_Test | LOVO model without Fiat Palio + PVS 7–9 in the app upload format, for an honest app test |

**How to run.** Open the notebooks in Colab, and upload `ml/c4/roughness/c4_pipeline.py` to
`MyDrive/PVS_Code/c4_pipeline.py`. The PVS data must be in `MyDrive/PVS1` … `PVS9`. Run the
notebooks in order: each one reads the outputs of the previous ones from Drive.

**Results of the app check (notebook 11 + `scripts/score_app_run.py`).** Macro-F1 on the unseen Fiat Palio:

| | Macro-F1 |
|---|---|
| Research pipeline | 0.833 |
| App adapter | 0.834 |
