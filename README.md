# Person Re-Identification Using Traditional Machine Learning Techniques

## Overview
This project implements a lightweight person re-identification (ReID) system using traditional machine learning techniques, achieving high accuracy on the Kaggle Person ReID dataset (684,287 images, 1,500 identities). The pipeline utilizes Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), and color histogram feature extractors, combined with Principal Component Analysis (PCA) for dimensionality reduction and K-Nearest Neighbors (KNN) for classification. The system is optimized for resource-constrained environments, making it suitable for real-time surveillance on edge devices.

## Key Features
- **Feature Extraction**: Implements HOG, SIFT, color histograms, and hybrid combinations (HOG+SIFT, HOG+color_hist, SIFT+color_hist, HOG+SIFT+color_hist).
- **Dimensionality Reduction**: Uses PCA to reduce feature dimensions to 100, 200, or 400 components for computational efficiency.
- **Classification**: Employs KNN with Euclidean and cosine similarity metrics, testing K values from 3 to 15.
- **Performance Metrics**: Evaluates cross-validation accuracy (up to 0.9703), training accuracy, Rank-1 (up to 0.974), and Rank-5 accuracies (up to 0.986).
- **Dataset**: Kaggle Person ReID dataset with 1,500 identities and diverse conditions (pose, illumination, camera angles).

## Results
- **Color-Based Features**: Achieved the highest cross-validation accuracies (0.9692–0.9703) with color histograms and hybrids.
- **HOG with Cosine Similarity**: Recorded strong ranking performance (Rank-1: 0.974, Rank-5: 0.986).
- **SIFT Limitations**: SIFT-based methods showed lower accuracies (0.3503–0.3704) due to reliance on local keypoints.

## Prerequisites
- Python 3.8
- Libraries: OpenCV 4.5, scikit-learn 0.24, NumPy 1.20
- Hardware: Tested on Apple M3 chip with 16GB RAM (8 cores)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhananjay-230/CSE623_ML_Group---Tech.git
   cd CSE623_ML_Group---Tech
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python==4.5.5.64 scikit-learn==0.24.2 numpy==1.20.3
   ```
3. Download the Kaggle Person ReID dataset and place it in the `data/` directory.

## Usage
1. **Prepare the Dataset**:
   - Resize images to 128x256 pixels.
   - Split dataset into 80% training (1,200 identities) and 20% validation (300 identities) using stratified sampling.

2. **Run the Pipeline**:
   ```bash
   python main.py --feature_method hog --k 3 --n_components 200
   ```
   - `--feature_method`: Options include `hog`, `sift`, `color_hist`, `hog_sift`, `hog_color_hist`, `sift_color_hist`, `hog_sift_color_hist`.
   - `--k`: KNN neighbors (e.g., 3, 5, 10, 15).
   - `--n_components`: PCA components (e.g., 100, 200, 400).

3. **Evaluate Results**:
   - Outputs include cross-validation accuracy, training accuracy, and CMC scores (Rank-1 and Rank-5 accuracies).

## Future Work
- Evaluate Rank-1 and Rank-5 accuracies for color-based feature extractors.
- Optimize KNN for real-time performance.
- Explore additional hybrid feature combinations.

## Contributors
- Rushi Moliya
- Shlok Shelat
- Dhananjay Kanjariya
- Shrey Salvi
- Purvansh Desai

## Acknowledgments
- Kaggle for providing the Person ReID dataset.
- Ahmedabad University for academic support.
