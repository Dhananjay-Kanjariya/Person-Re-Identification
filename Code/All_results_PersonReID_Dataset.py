import cv2
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Set dataset path
train_path = "/Users/shreysalvi/Desktop/VSCODE/COMPUTER_NETWORKS/ML/archive/train/"
batch_size = 5000

# Debug the path
print(f"Looking for training data at: {train_path}")
if not os.path.isdir(train_path):
    raise FileNotFoundError(f"Directory not found: {train_path}. Please check the path or download the dataset.")

# Function to load all images from subfolders and assign labels
def get_all_images(root_dir):
    image_paths = []
    labels = []
    for person_id in os.listdir(root_dir):  
        person_dir = os.path.join(root_dir, person_id)
        if os.path.isdir(person_dir):
            images = glob.glob(os.path.join(person_dir, "*.jpg"))  
            if not images:
                print(f"Warning: No .jpg images found in {person_dir}")
            image_paths.extend(images)
            labels.extend([str(person_id)] * len(images))  
    return image_paths, labels

# Load images and labels
train_images, image_labels = get_all_images(train_path)
if not train_images:
    raise ValueError("No images loaded. Check the dataset directory structure.")
print(f"✅ Loaded {len(train_images)} images from {len(set(image_labels))} people.")

# Split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, image_labels, test_size=1500, stratify=image_labels, random_state=42
)
print(f"Training set: {len(train_images)} images, Validation set: {len(val_images)} images")

# Feature extraction functions (unchanged)
def extract_hog_features(image):
    image_resized = cv2.resize(image, (128, 256))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hog_gray = hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    b, g, r = cv2.split(image_resized)
    hog_r = hog(r, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_g = hog(g, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_b = hog(b, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features = np.concatenate((hog_gray, hog_r, hog_g, hog_b))
    return features

def extract_sift_features(image):
    image_resized = cv2.resize(image, (128, 256))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    if descriptors is not None:
        sift_features = descriptors.mean(axis=0)
    else:
        sift_features = np.zeros(128)
    return sift_features

def extract_color_histogram_features(image):
    image_resized = cv2.resize(image, (128, 256))
    b, g, r = cv2.split(image_resized)
    hist_r = cv2.calcHist([r], [0], None, [64], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [64], [0, 256])
    hist_features = np.concatenate((hist_r, hist_g, hist_b)).flatten()
    return hist_features

def extract_features(image, method):
    if method == "hog":
        return extract_hog_features(image)
    elif method == "sift":
        return extract_sift_features(image)
    elif method == "color_hist":
        return extract_color_histogram_features(image)
    elif method == "hog_sift":
        hog_features = extract_hog_features(image)
        sift_features = extract_sift_features(image)
        return np.concatenate((hog_features, sift_features))
    elif method == "hog_color_hist":
        hog_features = extract_hog_features(image)
        hist_features = extract_color_histogram_features(image)
        return np.concatenate((hog_features, hist_features))
    elif method == "sift_color_hist":
        sift_features = extract_sift_features(image)
        hist_features = extract_color_histogram_features(image)
        return np.concatenate((sift_features, hist_features))
    elif method == "hog_sift_color_hist":
        hog_features = extract_hog_features(image)
        sift_features = extract_sift_features(image)
        hist_features = extract_color_histogram_features(image)
        return np.concatenate((hog_features, sift_features, hist_features))
    else:
        raise ValueError("Invalid feature extraction method")

feature_methods = ["hog", "sift", "color_hist", "hog_sift", "hog_color_hist", "sift_color_hist", "hog_sift_color_hist"]
similarity_metrics = ["euclidean", "cosine", "knn"]

def compute_knn_similarity(train_features, train_labels, val_features, val_labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(train_features, train_labels)
    similarities = []
    for i, val_feature in enumerate(val_features):
        val_feature = val_feature.reshape(1, -1)
        distances, indices = knn.kneighbors(val_feature, n_neighbors=k)
        neighbor_labels = [train_labels[idx] for idx in indices[0]]
        true_label = val_labels[i]
        similarity = sum(1 for label in neighbor_labels if label == true_label) / k
        similarities.append(similarity)
    return similarities

def compute_cmc(similarities, val_labels, train_labels, max_rank=20):
    cmc = np.zeros(max_rank)
    n_queries = len(val_labels)
    for i in range(n_queries):
        true_label = val_labels[i]
        sim_scores = similarities[i]
        ranked_indices = np.argsort(sim_scores)[::-1]
        for rank in range(max_rank):
            top_k_labels = [train_labels[idx] for idx in ranked_indices[:rank + 1]]
            if true_label in top_k_labels:
                cmc[rank] += 1
    cmc = cmc / n_queries
    return cmc

def evaluate_with_dimensionality_reduction(dim_reduction_method, method_name):
    results = {}
    plt.figure(figsize=(10, 6))
    for method in feature_methods:
        print(f"\n🔍 Evaluating feature extraction method: {method} with {method_name}")
        
        # Step 1: Extract features and fit IncrementalPCA incrementally
        train_feature_vectors = []
        train_valid_labels = []
        reducer = None
        
        print("Extracting features and fitting reducer for training set in batches...")
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            batch_features = []
            batch_valid_labels = []
            
            for img_path, label in zip(batch_images, batch_labels):
                image = cv2.imread(img_path)
                if image is None:
                    continue
                features = extract_features(image, method)
                batch_features.append(features)
                batch_valid_labels.append(label)
            
            batch_features = np.array(batch_features, dtype=np.float32)
            
            if dim_reduction_method == "pca":
                if reducer is None:
                    n_components = min(200, batch_features.shape[1])  # Ensure n_components <= features
                    reducer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
                reducer.partial_fit(batch_features)  # Incrementally fit the PCA
            else:  # t-SNE (limited to first batch for simplicity)
                if i == 0:
                    subsample_size = min(10000, len(batch_features))
                    subsample_indices = np.random.choice(len(batch_features), subsample_size, replace=False)
                    subsample_features = batch_features[subsample_indices]
                    reducer = TSNE(n_components=2, random_state=42, n_jobs=-1)
                    train_feature_vectors = reducer.fit_transform(subsample_features)
                    train_valid_labels = np.array(batch_valid_labels[:subsample_size])
                continue  # Skip further batches for t-SNE
            
            print(f"✅ Processed {i+len(batch_images)} images...")

        # Step 2: Transform training features after fitting
        if dim_reduction_method == "pca":
            train_feature_vectors = []
            train_valid_labels = []
            for i in range(0, len(train_images), batch_size):
                batch_images = train_images[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                batch_features = []
                batch_valid_labels = []
                for img_path, label in zip(batch_images, batch_labels):
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    features = extract_features(image, method)
                    batch_features.append(features)
                    
                    batch_valid_labels.append(label)
                batch_features = np.array(batch_features, dtype=np.float32)
                transformed_features = reducer.transform(batch_features)
                train_feature_vectors.append(transformed_features)
                train_valid_labels.extend(batch_valid_labels)
            train_feature_vectors = np.vstack(train_feature_vectors)
            train_valid_labels = np.array(train_valid_labels)
        
        print(f"📏 Original feature shape: ({len(train_images)}, {batch_features.shape[1]})")
        print(f"📉 Reduced feature shape: {train_feature_vectors.shape}")

        # Extract and transform validation features
        val_feature_vectors = []
        val_valid_labels = []
        print("Extracting features for validation set...")
        for img_path, label in zip(val_images, val_labels):
            image = cv2.imread(img_path)
            if image is None:
                continue
            features = extract_features(image, method)
            val_feature_vectors.append(features)
            val_valid_labels.append(label)
        val_feature_vectors = np.array(val_feature_vectors, dtype=np.float32)
        if dim_reduction_method == "pca":
            val_feature_vectors = reducer.transform(val_feature_vectors)
        else:
            combined_features = np.vstack((train_feature_vectors, val_feature_vectors))
            reducer = TSNE(n_components=2, random_state=42, n_jobs=-1)
            combined_reduced = reducer.fit_transform(combined_features)
            train_feature_vectors = combined_reduced[:len(train_valid_labels)]
            val_feature_vectors = combined_reduced[len(train_valid_labels):]
        val_valid_labels = np.array(val_valid_labels)

        # Evaluate with similarity metrics
        for metric in similarity_metrics:
            print(f"\n🔎 Testing with similarity metric: {metric}")
            if metric in ["euclidean", "cosine"]:
                if metric == "euclidean":
                    distances = euclidean_distances(val_feature_vectors, train_feature_vectors)
                    similarities = 1 / (1 + distances)
                else:
                    similarities = cosine_similarity(val_feature_vectors, train_feature_vectors)
                cmc = compute_cmc(similarities, val_valid_labels, train_valid_labels, max_rank=20)
                rank1_accuracy = cmc[0]
                rank5_accuracy = cmc[4]
                plt.plot(range(1, 21), cmc, label=f"{method}_{metric}")
                best_k = 1
                best_score = 0
                k_values = [3, 5]
                print("Testing different values of k for k-NN classifier...")
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                    scores = cross_val_score(knn, train_feature_vectors, train_valid_labels, cv=5)
                    avg_score = scores.mean()
                    print(f"📊 k = {k}, Cross-Validation Accuracy = {avg_score:.4f}")
                    if avg_score > best_score:
                        best_k = k
                        best_score = avg_score
                print(f"✅ Best k found: {best_k} with accuracy {best_score:.4f}")
                knn_final = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
                knn_final.fit(train_feature_vectors, train_valid_labels)
                train_pred = knn_final.predict(train_feature_vectors)
                train_acc = accuracy_score(train_valid_labels, train_pred)
                print(f"📊 Final Training Accuracy: {train_acc:.4f}")
            else:  # k-NN similarity
                similarities = compute_knn_similarity(train_feature_vectors, train_valid_labels, val_feature_vectors, val_valid_labels, k=5)
                cmc = compute_cmc(similarities, val_valid_labels, train_valid_labels, max_rank=20)
                rank1_accuracy = cmc[0]
                rank5_accuracy = cmc[4]
                plt.plot(range(1, 21), cmc, label=f"{method}_{metric}")
                best_k = 1
                best_score = 0
                k_values = [3, 5, 7]
                print("Testing different values of k for k-NN classifier...")
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
                    scores = cross_val_score(knn, train_feature_vectors, train_valid_labels, cv=5)
                    avg_score = scores.mean()
                    print(f"📊 k = {k}, Cross-Validation Accuracy = {avg_score:.4f}")
                    if avg_score > best_score:
                        best_k = k
                        best_score = avg_score
                print(f"✅ Best k found: {best_k} with accuracy {best_score:.4f}")
                knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
                knn_final.fit(train_feature_vectors, train_valid_labels)
                train_pred = knn_final.predict(train_feature_vectors)
                train_acc = accuracy_score(train_valid_labels, train_pred)
                print(f"📊 Final Training Accuracy: {train_acc:.4f}")
            key = f"{method}_{metric}"
            results[key] = {
                "best_k": best_k,
                "cross_val_accuracy": best_score,
                "training_accuracy": train_acc,
                "rank1_accuracy": rank1_accuracy,
                "rank5_accuracy": rank5_accuracy,
                "cmc": cmc,
                "reducer": reducer,
                "knn": knn_final
            }
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title(f"CMC Curves with {method_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"cmc_curves_{method_name.lower()}.png")
    plt.show()
    print(f"\n📊 Results Table with {method_name}:")
    print("| Feature Extractor         | Similarity Metric | Best k | Cross-Val Accuracy | Training Accuracy | Rank-1 Accuracy | Rank-5 Accuracy |")
    print("|---------------------------|-------------------|--------|--------------------|-------------------|-----------------|-----------------|")
    for key, result in results.items():
        feature_method, metric = key.split("_")
        print(f"| {feature_method:<24} | {metric:<16} | {result['best_k']:<6} | {result['cross_val_accuracy']:.4f}             | {result['training_accuracy']:.4f}          | {result['rank1_accuracy']:.4f}        | {result['rank5_accuracy']:.4f}        |")
    return results

# Run evaluation
results_pca = evaluate_with_dimensionality_reduction("pca", "PCA")
results_tsne = evaluate_with_dimensionality_reduction("tsne", "t-SNE")

# Find best combinations
best_combination_pca = max(results_pca, key=lambda x: results_pca[x]["rank1_accuracy"])
print(f"\n🏆 Best Combination with PCA: {best_combination_pca}")
print(f"Best Rank-1 Accuracy: {results_pca[best_combination_pca]['rank1_accuracy']:.4f}")

best_combination_tsne = max(results_tsne, key=lambda x: results_tsne[x]["rank1_accuracy"])
print(f"\n🏆 Best Combination with t-SNE: {best_combination_tsne}")
print(f"Best Rank-1 Accuracy: {results_tsne[best_combination_tsne]['rank1_accuracy']:.4f}")

# Save best model (PCA)
best_reducer = results_pca[best_combination_pca]["reducer"]
best_knn = results_pca[best_combination_pca]["knn"]
with open("best_knn_model.pkl", "wb") as f:
    pickle.dump(best_knn, f)
with open("best_pca_model.pkl", "wb") as f:
    pickle.dump(best_reducer, f)
with open("best_knn_labels.pkl", "wb") as f:
    pickle.dump(best_knn.classes_, f)
print("✅ Best model saved successfully!")
