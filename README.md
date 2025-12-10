# Radiomics-Based-Machine-Learning-Cancer-Immunotherapy
Radiomics-Based Machine Learning to Evaluate Immunotherapy Efficacy in Non-Small Cell Lung Cancer NSCLC Patients with Bone Metastases 

#Abstract
Purpose: Assessing treatment response in bone metastases from non–small cell lung cancer (NSCLC) remains a major clinical challenge, particularly for patients receiving immune checkpoint inhibitors (ICIs). Existing response criteria are not optimized for osseous disease, leading to inconsistent evaluation. This study aimed to develop and validate a radiomics-based machine learning (ML) framework to non-invasively distinguish immunotherapy response categories—progression, stable disease, and partial response—in NSCLC patients with bone metastases.
Approach: Chest CT scans from 99 NSCLC patients were analyzed before and during ICI therapy. Bone structures were automatically segmented using TotalSegmentator, and 1051 radiomic features were extracted per timepoint. Clinical variables were incorporated as optional features. Three ML classifiers—Random Forest, XGBoost, and Support Vector Machine—were trained using 5-fold cross-validation. A multistep feature selection pipeline (correlation filtering, mutual information, recursive feature elimination, and ReliefF ranking) was applied. Model performance was evaluated using AUC, F1-score, accuracy, sensitivity, and specificity, with additional statistical testing using Kruskal–Wallis, Mann–Whitney U, bootstrapping, and permutation analysis.
Results: Inter-rater agreement for radiological response categories was high (Cohen’s kappa = 0.91). Post-treatment radiomic features yielded the best performance. The Random Forest model achieved an AUC of 0.94, F1-score of 0.79, accuracy of 0.79, sensitivity of 0.80, and specificity of 0.83. Clinical features did not meaningfully improve performance. Models based on the largest lesion showed lower accuracy than those using the overall response.
Conclusions: Post-treatment CT radiomics captured therapy-induced skeletal changes and enabled differentiation of immunotherapy response categories in NSCLC bone metastases. These findings highlight radiomics as a non-invasive tool for response assessment and guiding personalized treatment strategies.


#=== PACKAGE VERSIONS ===
radiomics: v3.1.0
scikit-learn: 1.5.2
imblearn: 0.12.4
pingouin: 0.5.5
scipy: 1.13.1
xgboost: 2.1.3
skrebate: 0.62
numpy: 1.24.4
pandas: 2.2.2
shap: 0.47.2
