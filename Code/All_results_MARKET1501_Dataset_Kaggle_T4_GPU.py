# Install FAISS for GPU-accelerated similarity search
!pip install faiss-gpu

# Import FAISS
import faiss

# Your existing imports
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seed for reproducibility
torch.manual_seed(42)

import os
import glob
import cv2
import numpy as np
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import torch
import cupy as cp
from multiprocessing import Pool, cpu_count

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define dataset paths for the Market-1501 dataset
train_dir = "/kaggle/input/ml-project-group-tech/Market_dataset/train"
test_dir = "/kaggle/input/ml-project-group-tech/Market_dataset/test"

# Directory to save intermediate results
output_dir = "/kaggle/working/intermediate_results"  # Updated to your working directory
os.makedirs(output_dir, exist_ok=True)

print("Starting dataset loading...")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ------------------------------
# Utility Functions for Dataset Loading
# ------------------------------
def load_dataset(root_dir):
    print(f"Loading dataset from {root_dir}")
    image_paths = []
    labels = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        if os.path.isdir(person_path):
            imgs = glob.glob(os.path.join(person_path, "*.jpg"))
            if not imgs:
                print(f"Warning: No images in {person_path}")
            image_paths.extend(imgs)
            labels.extend([person_id] * len(imgs))
    return image_paths, labels

# ------------------------------
# Feature Extraction Functions
# ------------------------------
def extract_hog_features(image):
    image_resized = cv2.resize(image, (128, 256))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hog_gray = hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    b, g, r = cv2.split(image_resized)
    hog_r = hog(r, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_g = hog(g, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_b = hog(b, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return np.concatenate((hog_gray, hog_r, hog_g, hog_b))

def extract_sift_features(image):
    image_resized = cv2.resize(image, (128, 256))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    if descriptors is not None:
        return descriptors.mean(axis=0)
    return np.zeros(128)

def extract_color_histogram_features(image):
    image_resized = cv2.resize(image, (128, 256))
    b, g, r = cv2.split(image_resized)
    hist_r = cv2.calcHist([r], [0], None, [64], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [64], [0, 256])
    return np.concatenate((hist_r, hist_g, hist_b)).flatten()

def extract_features(args):
    image_path, method, idx, total = args
    if idx % 500 == 0 or idx == total - 1:
        print(f"Extracting features with method: {method} - Processing image {idx + 1}/{total}")
    image = cv2.imread(image_path)
    if image is None:
        return None
    if method == "hog":
        return extract_hog_features(image)
    elif method == "sift":
        return extract_sift_features(image)
    elif method == "color_hist":
        return extract_color_histogram_features(image)
    elif method == "hog+sift":
        hog_feat = extract_hog_features(image)
        sift_feat = extract_sift_features(image)
        return np.concatenate((hog_feat, sift_feat))
    elif method == "hog+color_hist":
        hog_feat = extract_hog_features(image)
        ch_feat = extract_color_histogram_features(image)
        return np.concatenate((hog_feat, ch_feat))
    elif method == "sift+color_hist":
        sift_feat = extract_sift_features(image)
        ch_feat = extract_color_histogram_features(image)
        return np.concatenate((sift_feat, ch_feat))
    elif method == "hog+sift+color_hist":
        hog_feat = extract_hog_features(image)
        sift_feat = extract_sift_features(image)
        ch_feat = extract_color_histogram_features(image)
        return np.concatenate((hog_feat, sift_feat, ch_feat))
    else:
        raise ValueError("Invalid feature extraction method")

# ------------------------------
# Batch Feature Extraction and PCA
# ------------------------------
def process_batch(image_paths, labels, method, max_components, pca=None):
    num_workers = cpu_count()
    print(f"Using {num_workers} CPU workers for parallel feature extraction")
    feat_list = []
    valid_labels = []
    
    with Pool(num_workers) as pool:
        args = [(path, method, i, len(image_paths)) for i, path in enumerate(image_paths)]
        feats = pool.map(extract_features, args)
    
    for feat, label in zip(feats, labels):
        if feat is not None:
            feat_list.append(feat)
            valid_labels.append(label)
    
    batch_feats = np.array(feat_list, dtype=np.float32)
    if batch_feats.size == 0:
        return None, None, pca
    
    # Incremental PCA
    if pca is None:
        n_components = min(max_components, batch_feats.shape[1])  # Ensure valid n_components
        pca = IncrementalPCA(n_components=n_components, batch_size=min(500, batch_feats.shape[0]))
    
    pca.partial_fit(batch_feats)
    reduced_feats = pca.transform(batch_feats)
    
    return reduced_feats, valid_labels, pca

def compute_features_in_batches(image_paths, labels, method, data_type, batch_size=1000, pca=None, max_components=200):
    print(f"Computing features in batches for {len(image_paths)} {data_type} images with method: {method}")
    all_reduced_feats = []
    all_labels = []
    
    # Calculate minimum batch size across all batches
    min_batch_size = min([min(batch_size, len(image_paths) - i) for i in range(0, len(image_paths), batch_size)])
    max_components = min(max_components, min_batch_size)  # Ensure n_components is valid for all batches
    
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_paths = image_paths[start:end]
        batch_labels = labels[start:end]
        print(f"Processing batch: {start} to {end}")
        
        # Pass max_components to ensure n_components is valid
        reduced_feats, valid_labels, pca = process_batch(batch_paths, batch_labels, method, max_components, pca)
        if reduced_feats is not None:
            all_reduced_feats.append(reduced_feats)
            all_labels.extend(valid_labels)
        
        # Save batch results as CSV with /train or /test in the name
        batch_df = pd.DataFrame(reduced_feats, columns=[f"feat_{i}" for i in range(reduced_feats.shape[1])])
        batch_df["label"] = valid_labels
        batch_file = os.path.join(output_dir, f"{method}_batch_{start}_{end}_reduced_{data_type}.csv")
        batch_df.to_csv(batch_file, index=False)
        print(f"Saved batch features to {batch_file}")
    
    return np.vstack(all_reduced_feats), all_labels, pca

# ------------------------------
# CMC Evaluation Function
# ------------------------------
def compute_cmc(similarity_matrix, test_labels, train_labels, max_rank=20):
    print("Computing CMC curve...")
    n_queries = similarity_matrix.shape[0]
    cmc = np.zeros(max_rank)
    for i in range(n_queries):
        sorted_idx = np.argsort(similarity_matrix[i])[::-1]
        for rank in range(max_rank):
            top_labels = [train_labels[idx] for idx in sorted_idx[:rank+1]]
            if test_labels[i] in top_labels:
                cmc[rank] += 1
                break
    return cmc / n_queries

# ------------------------------
# Similarity Computation in Batches
# ------------------------------
def compute_similarity_in_batches(test_feats_gpu, train_feats_gpu, metric, batch_size=100):
    n_test = test_feats_gpu.shape[0]
    n_train = train_feats_gpu.shape[0]  # Corrected to number of training samples
    sims = np.zeros((n_test, n_train), dtype=np.float32)  # Correct shape: (test_samples, train_samples)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        test_batch = test_feats_gpu[start:end]
        if metric == "euclidean":
            dists = cp.linalg.norm(test_batch[:, None] - train_feats_gpu, axis=2)  # Shape: (batch_size, n_train)
            sims[start:end] = cp.asnumpy(1 / (1 + dists))  # Shape: (batch_size, n_train)
        else:  # cosine
            sims[start:end] = cp.asnumpy(cp.dot(test_batch, train_feats_gpu.T) / 
                                         (cp.linalg.norm(test_batch, axis=1)[:, None] * 
                                          cp.linalg.norm(train_feats_gpu, axis=1)))
        cp.cuda.Stream.null.synchronize()  # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()  # Ensure memory is cleared
    return sims

# ------------------------------
# Main Evaluation Loop
# ------------------------------
print("Starting main evaluation loop...")
similarity_metrics = ["euclidean", "cosine"]
k_values = [3, 5, 9]  # Reduced set
results = {}
# feature_methods = ["hog"]  # Process one method at a time; change manually after each run
# feature_methods = ["hog+sift", "hog+color_hist", "sift+color_hist"]
feature_methods = ["hog", "sift", "color_hist", "hog+sift", "hog+color_hist", "sift+color_hist", "hog+sift+color_hist"]


# Load existing data or compute if not available
train_images, train_labels = load_dataset(train_dir)
test_images, test_labels = load_dataset(test_dir)

# Calculate global minimum batch size across train and test
min_train_batch_size = min([min(1000, len(train_images) - i) for i in range(0, len(train_images), 1000)])
min_test_batch_size = min([min(1000, len(test_images) - i) for i in range(0, len(test_images), 1000)])
global_max_components = min(200, min_train_batch_size, min_test_batch_size)

for feat_method in feature_methods:
    print(f"\nEvaluating feature extractor: {feat_method}")
    
    # Check if features are already computed
    train_feats_file = os.path.join(output_dir, f"{feat_method}_train_feats_reduced.csv")
    test_feats_file = os.path.join(output_dir, f"{feat_method}_test_feats_reduced.csv")
    pca_file = os.path.join(output_dir, f"{feat_method}_pca_model.pkl")
    
    if os.path.exists(train_feats_file) and os.path.exists(test_feats_file) and os.path.exists(pca_file):
        print("Loading precomputed features and PCA model...")
        train_df = pd.read_csv(train_feats_file)
        train_valid_labels = train_df["label"].tolist()
        train_feats_reduced = train_df.drop(columns=["label"]).values
        
        test_df = pd.read_csv(test_feats_file)
        test_valid_labels = test_df["label"].tolist()
        test_feats_reduced = test_df.drop(columns=["label"]).values
        
        with open(pca_file, "rb") as f:
            pca = pickle.load(f)
    else:
        # Process training features in batches
        train_feats_reduced, train_valid_labels, pca = compute_features_in_batches(train_images, train_labels, feat_method, "train", max_components=global_max_components)
        # Process test features using the same PCA model
        test_feats_reduced, test_valid_labels, _ = compute_features_in_batches(test_images, test_labels, feat_method, "test", pca=pca, max_components=global_max_components)
        
        # Save reduced features and PCA model
        train_df = pd.DataFrame(train_feats_reduced, columns=[f"feat_{i}" for i in range(train_feats_reduced.shape[1])])
        train_df["label"] = train_valid_labels
        train_df.to_csv(train_feats_file, index=False)
        
        test_df = pd.DataFrame(test_feats_reduced, columns=[f"feat_{i}" for i in range(test_feats_reduced.shape[1])])
        test_df["label"] = test_valid_labels
        test_df.to_csv(test_feats_file, index=False)
        
        with open(pca_file, "wb") as f:
            pickle.dump(pca, f)
        print(f"Saved reduced features and PCA model for {feat_method}")

    if train_feats_reduced.size == 0 or test_feats_reduced.size == 0:
        print("Skipping due to empty features.")
        continue

    # Move to GPU
    train_feats_gpu = cp.array(train_feats_reduced)
    test_feats_gpu = cp.array(test_feats_reduced)

    for metric in similarity_metrics:
        print(f"Processing similarity metric: {metric}")
        sims_cpu = compute_similarity_in_batches(test_feats_gpu, train_feats_gpu, metric)
        
        # Save similarity matrix as CSV
        sims_df = pd.DataFrame(sims_cpu)
        sims_df.to_csv(os.path.join(output_dir, f"{feat_method}_{metric}_sims.csv"), index=False)
        print(f"Saved similarity matrix for {feat_method}_{metric}")

        cmc_curve = compute_cmc(sims_cpu, test_valid_labels, train_valid_labels, max_rank=20)
        rank1_acc = cmc_curve[0]
        rank5_acc = cmc_curve[4]
        
        # Save CMC curve as CSV
        pd.DataFrame(cmc_curve, columns=["cmc"]).to_csv(os.path.join(output_dir, f"{feat_method}_{metric}_cmc_curve.csv"), index=False)
        print(f"Saved CMC curve for {feat_method}_{metric}")

        for k in k_values:
            print(f"Evaluating k={k}")
            knn_cv = KNeighborsClassifier(n_neighbors=k, metric=metric)
            cv_scores = cross_val_score(knn_cv, train_feats_reduced, train_valid_labels, cv=5)
            cv_acc = cv_scores.mean()
            print(f"Cross-validation completed: {cv_acc:.4f}")

            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(train_feats_reduced, train_valid_labels)
            train_pred = knn.predict(train_feats_reduced)
            train_acc = accuracy_score(train_valid_labels, train_pred)
            print(f"Training accuracy: {train_acc:.4f}")

            test_pred = knn.predict(test_feats_reduced)
            test_acc = accuracy_score(test_valid_labels, test_pred)
            precision = precision_score(test_valid_labels, test_pred, average='macro', zero_division=0)
            recall = recall_score(test_valid_labels, test_pred, average='macro', zero_division=0)
            f1 = f1_score(test_valid_labels, test_pred, average='macro', zero_division=0)
            print(f"Test metrics - Accuracy: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"CMC - Rank-1: {rank1_acc:.4f}, Rank-5: {rank5_acc:.4f}")

            # Save predictions and k-NN model
            pd.DataFrame(train_pred, columns=["pred"]).to_csv(os.path.join(output_dir, f"{feat_method}_{metric}_k{k}_train_pred.csv"), index=False)
            pd.DataFrame(test_pred, columns=["pred"]).to_csv(os.path.join(output_dir, f"{feat_method}_{metric}_k{k}_test_pred.csv"), index=False)
            with open(os.path.join(output_dir, f"{feat_method}_{metric}_k{k}_knn_model.pkl"), "wb") as f:
                pickle.dump(knn, f)
            print(f"Saved predictions and k-NN model for {feat_method}_{metric}_k{k}")

            key = f"{feat_method}_PCA_{metric}_k{k}"
            results[key] = {
                "Feature Extractor": feat_method,
                "Reduction": "PCA",
                "Similarity": metric,
                "k": k,
                "CV Accuracy": cv_acc,
                "Train Acc": train_acc,
                "Rank-1 Acc": rank1_acc,
                "Rank-5 Acc": rank5_acc,
                "Test Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            print(f"Completed: {key}")

    # Clear GPU memory
    del train_feats_gpu, test_feats_gpu
    cp.get_default_memory_pool().free_all_blocks()

# Save results for this feature method as CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv(os.path.join(output_dir, f"results_{feature_methods[0]}.csv"), index=True)
print(f"Saved results for {feature_methods[0]}")


import os
import pandas as pd

output_dir = "intermediate_results"
feature_methods = ["hog", "sift", "color_hist", "hog+sift", "hog+color_hist", "sift+color_hist"]
all_results = {}

for feat_method in feature_methods:
    result_file = os.path.join(output_dir, f"results_{feat_method}.csv")
    if os.path.exists(result_file):
        df = pd.read_csv(result_file, index_col=0)
        all_results.update(df.to_dict(orient="index"))

# Print the table
header = ["Feature Extractor", "Reduction", "Similarity", "k", "CV Accuracy", "Train Acc", 
          "Rank-1 Acc", "Rank-5 Acc", "Test Accuracy", "Precision", "Recall", "F1 Score"]
print("\n" + "-"*120)
print("| " + " | ".join(f"{h:^15}" for h in header) + " |")
print("-"*120)
for key, res in all_results.items():
    print("| " + " | ".join(f"{str(res[h]):^15}" if h in res else f"{'':^15}" for h in header) + " |")
print("-"*120)

# Save combined results as CSV
combined_df = pd.DataFrame.from_dict(all_results, orient="index")
combined_df.to_csv(os.path.join(output_dir, "market_results_combined.csv"), index=True)
print("Combined results saved.")
