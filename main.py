# %%
# === 1. IMPORTS ===
# Standard Library Imports
import os
import logging
import sys
import re
import warnings

# Data Handling and Numerics
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, ttest_ind

# Radiomics
import radiomics
from radiomics import featureextractor, getFeatureClasses
import glob

# Machine Learning & Metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, recall_score, confusion_matrix, cohen_kappa_score, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
from skrebate import ReliefF
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import pickle
from pingouin import intraclass_corr
from matplotlib import pyplot as plt

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# %%
# === 2. ENVIRONMENT AND LOGGING SETUP ===
# Print Package Versions for Reproducibility
print("\n=== PACKAGE VERSIONS ===")
# Radiomics (PyRadiomics)
try:
    import radiomics
    print("radiomics:", radiomics.__version__)
except:
    pass
# sklearn
try:
    import sklearn
    print("scikit-learn:", sklearn.__version__)
except:
    pass
# imblearn
try:
    import imblearn
    print("imblearn:", imblearn.__version__)
except:
    pass
# pingouin
try:
    import pingouin
    print("pingouin:", pingouin.__version__)
except:
    pass
# scipy
try:
    import scipy
    print("scipy:", scipy.__version__)
except:
    pass
# xgboost
try:
    import xgboost
    print("xgboost:", xgboost.__version__)
except:
    pass
# skrebate
try:
    import skrebate
    print("skrebate:", skrebate.__version__)
except:
    pass
# Other packages using __version__
others = [np, pd, shap]
for mod in others:
    try:
        print(f"{mod.__name__}:", mod.__version__)
    except AttributeError:
        pass

# PyRadiomics Logger Setup
logger = radiomics.logger
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
# === 3. UTILITY FUNCTIONS ===

def disagreement(radiologist1, radiologist2):
    """Calculates and returns a DataFrame of case IDs where two radiologists disagree."""
    # Create a DataFrame for better visualization
    data = {
        'Case ID': Record_ID,  # Assumes Record_ID is globally available from Rdy_1/Rdy_2 loading
        'Radiologist 1': radiologist1,
        'Radiologist 2': radiologist2,
        'Agreement': [r1 == r2 for r1, r2 in zip(radiologist1, radiologist2)]
    }
    df = pd.DataFrame(data)
    # Filter cases where disagreement occurred
    disagreements = df[df['Agreement'] == False]
    return disagreements

# %%
def get_data_directories(directory_path):
    """
    Traverses a directory and returns paths to folders that do not start with a number
    and contain .gz or .inp files (indicating NIfTI or mask data).
    """
    # Regular expression to check if a name starts with a number
    pattern = re.compile(r"^\d")
    # List to store folders that don't start with a number
    non_numeric_folders = []
    # Traverse the directory and subdirectories
    if os.path.exists(directory_path):
        for root, dirs, _ in os.walk(directory_path):
            for folder_name in dirs:
                if not pattern.match(folder_name):
                    folder_path = os.path.join(root, folder_name)
                    # Check for NIfTI/inp files inside the folder
                    files = os.listdir(folder_path)
                    if any(f.endswith('.gz') or f.endswith('.inp') for f in files):
                        non_numeric_folders.append(folder_path)
    else:
        print(f"The directory '{directory_path}' does not exist.")
        return []

    if non_numeric_folders:
        print(f"Found {len(non_numeric_folders)} relevant folders.")
    else:
        print("No relevant folders found.")
    return non_numeric_folders

# %%
def find_corresponding_pre_post(post_address, pre_address):
    """
    Matches pre-treatment and post-treatment segmentations based on the 3-digit patient ID
    embedded in the path (assuming the format: D:\Bone metastases\converted_nifti_Seg\<ID>_<SCAN_TYPE>_post...).
    """
    corresp_seg = []
    post_address = np.asarray(post_address)
    pre_address = np.asarray(pre_address)

    for i in range(post_address.shape[0]):
        # Extract ID from the path (assuming structure: index 3 is the ID_SCAN_TYPE folder)
        directories_post = post_address[i, 1].split(os.sep)
        third_directory_post = directories_post[3][0:3]

        for j in range(pre_address.shape[0]):
            directories_pre = pre_address[j, 1].split(os.sep)
            third_directory_pre = directories_pre[3][0:3]

            if third_directory_pre == third_directory_post:
                corresp_seg.append([pre_address[j], post_address[i]])
                # Break inner loop once a match is found
                break
    return corresp_seg

# %%
# === 4. FEATURE REDUCTION FUNCTIONS ===
# -------------------------------
# Step 1: Handling Multicollinearity
# -------------------------------
def handling_multicollinearity(features, feature_names):
    print("\n--- Step 1: Handling Multicollinearity ---")
    df_features = pd.DataFrame(features, columns=feature_names)
    correlation_matrix = df_features.corr().values
    threshold = 0.99
    
    # Calculate the absolute correlation matrix
    correlation_matrix = np.abs(correlation_matrix)
    
    # Use a set to store indices (columns) to drop
    features_to_drop_indices = set()
    
    # Iterate over the upper triangle (k=1)
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if correlation_matrix[i, j] > threshold:
                # Always drop the feature with the higher index (to simplify tracking)
                features_to_drop_indices.add(j)

    # Get the names and indices of the remaining features
    features_to_drop_list = sorted(list(features_to_drop_indices), reverse=True)
    
    # Drop columns by index
    reduced_features = df_features.drop(df_features.columns[features_to_drop_list], axis=1)
    
    # Update feature names
    updated_feature_names = reduced_features.columns.tolist()

    print(f"Original number of features: {df_features.shape[1]}")
    print(f"Number of features removed: {len(features_to_drop_list)}")
    print(f"Number of features after reduction: {reduced_features.shape[1]}")
    return reduced_features, updated_feature_names

# -------------------------------
# Step 2: Univariate Statistical Selection (Mutual Information)
# -------------------------------
def univariate_statistical_selection(features, feature_names, output_labels):
    print("\n--- Step 2: Univariate Statistical Selection (Mutual Information) ---")
    features_array = features.values if isinstance(features, pd.DataFrame) else features
    
    # Select top 50% of features
    k_features = int(features_array.shape[1] / 2)
    mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    mutual_info_selector.fit(features_array, output_labels)
    
    # Get scores and selected feature names/data
    mutual_info_scores = mutual_info_selector.scores_
    support_mask = mutual_info_selector.get_support()
    selected_feature_names = np.array(feature_names)[support_mask]
    
    # Create the reduced feature set
    reduced_features_univariate = features.loc[:, selected_feature_names]

    # Display top features
    top_univariate_features = pd.DataFrame({
        "Feature": feature_names,
        "Score": mutual_info_scores
    }).sort_values(by="Score", ascending=False).head(10)
    print("Top 10 Mutual Information Scores:")
    print(top_univariate_features)

    print(f"Number of features retained after Univariate Selection: {len(selected_feature_names)}")
    return reduced_features_univariate, selected_feature_names

# -------------------------------
# Step 3: Model-Based Feature Importance (SHAP with Random Forest)
# -------------------------------
def shap_feature_importance(features, feature_names, output_labels):
    print("\n--- Step 3: Feature Importance Using SHAP with Random Forest ---")
    features_array = features.values if isinstance(features, pd.DataFrame) else features
    
    # Fit Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(features_array, output_labels)
    
    # Create SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(features_array)
    
    # For multiclass, take mean of absolute SHAP across all classes
    if isinstance(shap_values, list):
        # shap_values is a list of arrays, one per class
        # Calculate mean(|SHAP|) for each feature for each class, then average across classes
        shap_values_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Binary classification case
        shap_values_mean = np.abs(shap_values).mean(axis=0)
        
    # Create DataFrame of SHAP importances
    shap_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": shap_values_mean
    }).sort_values(by="Importance", ascending=False)
    
    # Retain top 50% of features
    n_top_features = int(len(shap_importance_df) / 2)
    selected_features_shap_df = shap_importance_df.iloc[:n_top_features]
    
    # Select those features
    selected_feature_names = selected_features_shap_df["Feature"].values
    reduced_features_shap = features.loc[:, selected_feature_names]
    
    print("Top 10 features by SHAP:")
    print(selected_features_shap_df.head(10))
    print(f"Number of features retained after SHAP: {len(selected_feature_names)}")
    return reduced_features_shap, selected_feature_names

# -------------------------------
# Step 4: Recursive Feature Elimination (RFE)
# -------------------------------
def recursive_feature_elimination(features, feature_names, output_labels):
    print("\n--- Step 4: Recursive Feature Elimination (RFE) ---")
    features_array = features.values if isinstance(features, pd.DataFrame) else features
    
    # Use Logistic Regression for RFE
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Select top 50% of features
    n_features_to_select = int(features_array.shape[1] / 2)
    rfe_selector = RFE(estimator=logistic_model, n_features_to_select=n_features_to_select)
    rfe_selector.fit(features_array, output_labels)
    
    # Get selected feature names
    support_mask = rfe_selector.get_support()
    selected_feature_names = np.array(feature_names)[support_mask]
    
    # Retain only selected columns
    reduced_features_RFE = features.loc[:, selected_feature_names]
    
    print(f"Original features shape for RFE: {features_array.shape}")
    print(f"Number of features retained after RFE: {len(selected_feature_names)}")
    return reduced_features_RFE, selected_feature_names

# %%
def reduce_features(features, feature_names, output_labels):
    """
    Applies the full 4-step feature reduction pipeline.
    """
    print("\n=======================================================")
    print(f"Starting Feature Reduction. Initial Features: {features.shape[1]}")
    print("=======================================================")

    # 1. Multicollinearity Handling
    features_1, names_1 = handling_multicollinearity(features, feature_names)

    # 2. Univariate Statistical Selection (MI)
    features_2, names_2 = univariate_statistical_selection(features_1, names_1, output_labels)
    
    # 3. Model-Based Feature Importance (SHAP/RF)
    features_3, names_3 = shap_feature_importance(features_2, names_2, output_labels)

    # 4. Recursive Feature Elimination (RFE/LogReg)
    features_4, names_4 = recursive_feature_elimination(features_3, names_3, output_labels)

    print("\n=======================================================")
    print(f"Feature Reduction Complete. Final Features: {features_4.shape[1]}")
    print("=======================================================")
    
    # Final scaling (StandardScaler) and SMOTE (Oversampling)
    # The original code only defined the functions, but if you were to complete the code:
    
    # Standardize the final features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_4.values)
    
    # Apply SMOTE (requires defining max_count based on class distribution)
    # Find max_count for SMOTE target
    label_counts = Counter(output_labels)
    max_count = max(label_counts.values())
    target_samples = {label: max_count for label in np.unique(output_labels)}
    
    # Initialize SMOTE with the specified targets
    sm = SMOTE(sampling_strategy=target_samples, random_state=42)
    features_smote, labels_smote = sm.fit_resample(features_scaled, output_labels)

    print(f"Features after Scaling and SMOTE: {features_smote.shape}")
    print(f"Labels after SMOTE balancing: {Counter(labels_smote)}")

    return features_smote, labels_smote, names_4

# %%
# === 5. DATA LOADING: CLINICAL & RESPONSE DATA ===
# Load Rater 1 Data (Pete MBD Imaging Review)
file_path_rdy1 = 'D:\\Bone metastases\\dataset\\Pete MBD Imaging Review.xlsx'
Rdy_1 = pd.read_excel(file_path_rdy1, engine='openpyxl')
Record_ID = Rdy_1['Record ID']
Largest_lesion_response_Rdy_1 = Rdy_1['Pete Largest lesion response']
Overall_response_Rdy_1 = Rdy_1['Pete Overall Response']
RT_info = pd.Series(Rdy_1['RT Location 1']).tolist()
RT_status = pd.DataFrame(RT_info, columns=["Values"])
# Binarize RT status: NaN -> "no", others -> "yes"
RT_status["Result"] = RT_status["Values"].apply(lambda x: "no" if pd.isna(x) else "yes")
print(f"RT Status records loaded: {len(RT_status['Result'].tolist())}")

# %%
# Load Rater 2 Data (Hila Yashar)
file_path_rdy2 = 'D:\\Bone metastases\\dataset\\Results bone lesions_Hila Yashar.xlsx'
Rdy_2 = pd.read_excel(file_path_rdy2, engine='openpyxl')
# Record_ID is overwritten here, but assumed to be the same list as Rdy_1 for Kappa
Largest_lesion_response_Rdy_2 = Rdy_2['Largest Lesion Response']
Overall_response_Rdy_2 = Rdy_2['Overall Response']

# %%
# === 6. INTER-RATER AGREEMENT (KAPPA) ===
kappa_Overall_response = cohen_kappa_score(Overall_response_Rdy_1.tolist(), Overall_response_Rdy_2.tolist())
kappa_Largest_lesion = cohen_kappa_score(Largest_lesion_response_Rdy_1.tolist(), Largest_lesion_response_Rdy_2.tolist())
print('Inter-Rater Agreement:')
print(f'  Kappa (Overall Response) = {kappa_Overall_response:.3f}')
print(f'  Kappa (Largest Lesion) = {kappa_Largest_lesion:.3f}')

# %%
# Display Disagreements
kappa_Overall_response_disagreement = disagreement(Overall_response_Rdy_1.tolist(), Overall_response_Rdy_2.tolist())
print('\nOverall_response_disagreement:')
print(kappa_Overall_response_disagreement)

# %%
kappa_Largest_lesion_disagreement = disagreement(Largest_lesion_response_Rdy_1.tolist(), Largest_lesion_response_Rdy_2.tolist())
print('\nLargest_lesion_disagreement:')
print(kappa_Largest_lesion_disagreement)

# %%
# === 7. FILE PATH GATHERING ===
seg_paths = get_data_directories("D:\Bone metastases\converted_nifti_Seg")
img_paths = get_data_directories("D:\Bone metastases\converted_nifti")

# %%
# Count scan types (for verification)
abd_pelvis_count = sum(1 for address in seg_paths if "Abd_Pelvis" in address)
chest_count = sum(1 for address in seg_paths if "Chest" in address)
abd_count = sum(1 for address in seg_paths if "\\Abd\\" in address)
print(f"Scan Count Check: Abd_Pelvis={abd_pelvis_count}, Chest={chest_count}, Abd={abd_count}")

# %%
# Get specific file paths for feature extraction (all_bones.nii.gz)
seg_path_all_bones = []
img_path_all_bones = []
for i in range(0, len(seg_paths)):
    path_img = img_paths[i] + '\\'
    path_mask = seg_paths[i] + '\\'
    # Use glob to find the specific files
    try:
        imageName = glob.glob(path_img + '*.nii.gz')[0]
        maskName = glob.glob(path_mask + 'all_bones.nii.gz')[0]
        seg_path_all_bones.append(maskName)
        img_path_all_bones.append(imageName)
    except IndexError:
        print(f"Warning: Could not find matching image/mask for path index {i}. Skipping.")

# %%
# === 8. SEGREGATE PRE/POST AND BODY REGIONS ===
post_address = []
pre_address = []
post_address_chest = []
post_address_abd_pelvic = []
post_address_abd = []
pre_address_chest = []
pre_address_abd_pelvic = []
pre_address_abd = []

for i in range(len(seg_path_all_bones)):
    path = seg_path_all_bones[i]
    if '_post' in path:
        post_address.append([i, path])
        if "Chest" in path:
            post_address_chest.append([i, path])
        elif "Abd_Pelvis" in path:
            post_address_abd_pelvic.append([i, path])
        elif "\\Abd\\" in path:
            post_address_abd.append([i, path])

    elif '_pre' in path:
        pre_address.append([i, path])
        if "Chest" in path:
            pre_address_chest.append([i, path])
        elif "Abd_Pelvis" in path:
            pre_address_abd_pelvic.append([i, path])
        elif "\\Abd\\" in path:
            pre_address_abd.append([i, path])

print(f"Found: post={len(post_address)}, pre={len(pre_address)}")
print(f"Chest: post={len(post_address_chest)}, pre={len(pre_address_chest)}")

# %%
# Match pre and post pairs by Patient ID
chest_address = find_corresponding_pre_post(post_address_chest, pre_address_chest)
abd_pelvic_address = find_corresponding_pre_post(post_address_abd_pelvic, pre_address_abd_pelvic)
abd_address = find_corresponding_pre_post(post_address_abd, pre_address_abd)

# Extract indices and addresses for the CHEST cohort (as chosen in the original script)
pre_chest_indx = np.asarray(chest_address)[:, 0, 0].astype(int)
post_chest_indx = np.asarray(chest_address)[:, 1, 0].astype(int)
pre_chest_address = np.asarray(chest_address)[:, 0, 1]
post_chest_address = np.asarray(chest_address)[:, 1, 1]

# Set the final cohort (was originally set to only chest)
pre_all_indx = pre_chest_indx
post_all_indx = post_chest_indx
pre_all_address = pre_chest_address
post_all_address = post_chest_address

# %%
print("Example pre-address (Chest Cohort):", pre_all_address[0])

# %%
# === 9. RADIOMICS FEATURE LOADING AND PROCESSING ===
with open('D:\\Bone metastases\\features\\' + 'features.npy', 'rb') as f:
    features_all_np = np.load(f, allow_pickle=True)

# %%
# Get feature names (Original, LOG, Wavelet)
feature_names_original = list(sorted(filter(lambda k: k.startswith("original_"), features_all_np[0])))
feature_names_LOG = list(sorted(filter(lambda k: k.startswith("log-"), features_all_np[0])))
feature_names_wavelet = list(sorted(filter(lambda k: k.startswith("wavelet-"), features_all_np[0])))
feature_names = feature_names_original + feature_names_LOG + feature_names_wavelet

# Extract feature values into a 2D NumPy array
samples = np.zeros((len(features_all_np), len(feature_names)))
for case_id in range(len(features_all_np)):
    a = np.array([features_all_np[case_id][feature_name] for feature_name in feature_names])
    samples[case_id, :] = a

# Handle potential NaNs
samples = np.nan_to_num(samples)
print(f"Total extracted feature samples (pre/post combined): {samples.shape}")

# %%
# Separate pre and post features based on matched indices
samples_pre = samples[pre_all_indx, :]
samples_post = samples[post_all_indx, :]
print(f"Matched Pre-treatment features shape: {samples_pre.shape}")
print(f"Matched Post-treatment features shape: {samples_post.shape}")

# %%
# === 10. CLINICAL DATA MAPPING AND NORMALIZATION ===
# Load additional clinical data (Sex, Days from ICI, response labels, Age, BMI)
file_path_modified = 'D:\\Bone metastases\\dataset\\Pete MBD Imaging Review_modified-ver02.xlsx'
data = pd.read_excel(file_path_modified)

with open('D:\\Bone metastases\\dataset\\sex_data.npy', 'rb') as f:
    sex_data = np.load(f)
with open('D:\\Bone metastases\\dataset\\num_days_from_ICI_to_Post.npy', 'rb') as f:
    ID_and_num_days = np.load(f)
ids = ID_and_num_days[:, 0].astype(int)

# Match Response, RT, Age, and BMI to the selected (Chest) cohort IDs
corresp_index_labels = []
for i in range(pre_all_address.shape[0]):
    directories = pre_all_address[i].split(os.sep)
    patient_id = int(directories[3][0:3]) # Get patient ID from path
    
    # Find index in the clinical data file
    indx_data = np.where(data["Record ID"] == patient_id)[0][0]
    
    # RT status is pulled from the initial RT_status DataFrame (indexed to Rdy_1)
    indx_rt = np.where(Rdy_1["Record ID"] == patient_id)[0][0]
    
    corresp_index_labels.append([
        patient_id,
        data["Pete Overall Response"][indx_data],
        data["Pete Largest lesion response"][indx_data],
        np.asarray(RT_status)[indx_rt, 1], # RT status (yes/no)
        data['Age at Baseline Imaging'][indx_data],
        data["BMI"][indx_data]
    ])

corresp_index_labels = np.asarray(corresp_index_labels)

# Match Sex data to the selected cohort IDs
corresp_sex_data = []
for i in range(corresp_index_labels.shape[0]):
    patient_id = int(corresp_index_labels[i, 0])
    indx_sex = np.where(ids == patient_id)[0][0]
    corresp_sex_data.append(sex_data[indx_sex])

inputs = np.column_stack((corresp_index_labels, np.asarray(corresp_sex_data)))

radiotherapy_info = inputs[:, 3]
sex_info = inputs[:, 6]
age_info = inputs[:, 4]
bmi_info = inputs[:, 5]
overall_response_info = inputs[:, 1]
largest_response_info = inputs[:, 2]

# Normalize Age and BMI
min_val_age = np.min(age_info.astype(int))
max_val_age = np.max(age_info.astype(int))
normalized_age = (age_info.astype(int) - min_val_age) / (max_val_age - min_val_age)

# Find non-NaN min/max for BMI (assuming the smallest non-zero value is the true minimum)
bmi_floats = bmi_info.astype(float)
valid_bmi = bmi_floats[~np.isnan(bmi_floats)]
min_val_bmi = np.min(valid_bmi)
max_val_bmi = np.max(valid_bmi)
normalized_bmi = (bmi_floats - min_val_bmi) / (max_val_bmi - min_val_bmi)
normalized_bmi = np.nan_to_num(normalized_bmi) # Replace NaN in BMI with 0 after normalization

# %%
# === 11. LABEL ENCODING AND FEATURE SET CREATION ===
ground_truth = np.copy(largest_response_info)

# Map labels to integers
label_mapping = {"partial response": 0, "stable": 1, "progression": 2}
output_labels = np.array([label_mapping[label] for label in ground_truth])
RT_mapping = {"no": 0, "yes": 1}
RT_status_binary = np.array([RT_mapping[label] for label in radiotherapy_info])
Sex_mapping = {"F": 0, "M": 1}
Sex_status_binary = np.array([Sex_mapping[label] for label in sex_info])

# Base Radiomics features (Pre and Post)
pre_treatment_features = samples_pre
post_treatment_features = samples_post

# Count and report class imbalance
label_counts = Counter(output_labels)
total_labels = len(output_labels)
imbalance_ratio = max(label_counts.values()) / min(label_counts.values())
print(f"\nLabel Counts (Largest Lesion Response): {label_counts}")
print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")

# Create combined feature sets with clinical variables
# FUSION: PRE + POST Radiomics
all_features = np.concatenate((pre_treatment_features, post_treatment_features), axis=1)

# Feature Sets + RT
pre_features_RT = np.concatenate((pre_treatment_features, RT_status_binary.reshape(-1, 1)), axis=1)
post_features_RT = np.concatenate((post_treatment_features, RT_status_binary.reshape(-1, 1)), axis=1)
all_features_RT = np.concatenate((all_features, RT_status_binary.reshape(-1, 1)), axis=1)

# Feature Sets + Sex
pre_features_Sex = np.concatenate((pre_treatment_features, Sex_status_binary.reshape(-1, 1)), axis=1)
all_features_Sex = np.concatenate((all_features, Sex_status_binary.reshape(-1, 1)), axis=1)

# Feature Sets + Age (Normalized)
all_features_age = np.concatenate((all_features, normalized_age.reshape(-1, 1)), axis=1)

# Feature Sets + BMI (Normalized)
all_features_bmi = np.concatenate((all_features, normalized_bmi.reshape(-1, 1)), axis=1)

# Feature Names (for tracking)
base_pre_names = [f"pre_{name}" for name in feature_names]
base_post_names = [f"post_{name}" for name in feature_names]

all_feature_names = base_pre_names + base_post_names
all_feature_names_RT = all_feature_names + ['RT']
all_feature_names_Sex = all_feature_names + ['Sex']
all_feature_names_age = all_feature_names + ['Age']
all_feature_names_bmi = all_feature_names + ['BMI']

# %%
# === 12. EXECUTION EXAMPLE: APPLY FEATURE REDUCTION (Fusion + RT) ===
print("\n--- Example Execution of Feature Reduction Pipeline ---")
features_to_reduce = all_features_RT
names_to_reduce = all_feature_names_RT

# Use the function defined in Section 4
final_features, final_labels, final_names = reduce_features(
    features=features_to_reduce, 
    feature_names=names_to_reduce, 
    output_labels=output_labels
)

# You would now proceed to train models (e.g., K-Fold CV with XGBoost) using final_features and final_labels.
# %%
# === 13. MODEL TRAINING AND EVALUATION SETUP ===
def evaluate_model_cv(X, y, feature_names, model_class, n_splits=5, random_state=42):
    """
    Performs Stratified K-Fold Cross-Validation and evaluates classification metrics.

    Args:
        X (np.ndarray): The feature matrix (scaled and SMOTEd).
        y (np.ndarray): The label vector (SMOTEd).
        feature_names (list): The final list of selected feature names.
        model_class: The scikit-learn compatible classifier class (e.g., XGBClassifier).
        n_splits (int): Number of folds for CV.
    
    Returns:
        dict: A dictionary containing averaged metrics across all folds.
    """
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for metrics across folds
    metrics = {
        'accuracy': [],
        'f1_macro': [],
        'auc_ovr': [],
        'sensitivity': [], # Recall for class 0 (e.g., Partial Response)
        'specificity': []  # TN / (TN + FP)
    }
    
    # Binarizer for AUC calculation (multiclass OVR: One-vs-Rest)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    
    print(f"\nStarting {n_splits}-Fold Stratified CV with {model_class.__name__}...")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_test_bin = y_bin[test_index]

        # Initialize and train the model
        model = model_class(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
        model.fit(X_train, y_train)
        
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate AUC (One-vs-Rest for multiclass)
        try:
            auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
        except ValueError:
            auc_ovr = np.nan # Handle cases where roc_auc_score fails
        
        # Calculate Sensitivity/Specificity for the first class (e.g., Response=0)
        # This requires focusing the CM on the class of interest
        TP = cm[0, 0]
        FN = np.sum(cm[0, 1:])
        FP = np.sum(cm[1:, 0])
        TN = np.sum(cm[1:, 1:]) # Sum of all non-class 0 predictions

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Store metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        metrics['auc_ovr'].append(auc_ovr)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)

        print(f"Fold {fold+1}: Accuracy={metrics['accuracy'][-1]:.4f}, AUC-OVR={metrics['auc_ovr'][-1]:.4f}")

    # Calculate and print averaged results
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}

    print("\n--- CROSS-VALIDATION RESULTS (Mean ± Std) ---")
    print(f"Model: {model_class.__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Accuracy: {avg_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}")
    print(f"F1-Macro: {avg_metrics['f1_macro']:.3f} ± {std_metrics['f1_macro']:.3f}")
    print(f"AUC-OVR: {avg_metrics['auc_ovr']:.3f} ± {std_metrics['auc_ovr']:.3f}")
    print(f"Sensitivity (Class 0): {avg_metrics['sensitivity']:.3f} ± {std_metrics['sensitivity']:.3f}")
    print(f"Specificity (Class 0): {avg_metrics['specificity']:.3f} ± {std_metrics['specificity']:.3f}")

    return avg_metrics

# %%
# === 14. EXECUTION OF MODEL EVALUATION ===

# Re-use the outputs from the Feature Reduction step (Section 12)
# X_final (features), y_final (labels), names_final (feature names)
X_final = final_features
y_final = final_labels
names_final = final_names

# Run the cross-validation using XGBoost
evaluation_results = evaluate_model_cv(
    X=X_final, 
    y=y_final, 
    feature_names=names_final, 
    model_class=XGBClassifier, 
    n_splits=5 # Commonly used number of folds
)

# %%