# Russian-Traffic-Sign-classification-methods
This repository implements and compares quality and zero-shot capabilities of NN classification methods from softmax classifiers to feature-learning models

## Overview
This project implements and compares various deep learning methods for a **multi-label classification task** using the cropped [**Russian Traffic Road Sign Dataset (RTSD)**](https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html).  

The key feature of this dataset is its **train-test split**:
- **Frequent signs**: Present in both train and test sets.
- **Rare signs**: Present only in the test set, **absent in the train set** — allowing evaluation of **zero-shot** classification approaches.

---

## Approaches Implemented

### 1. **Classic Softmax Classification**
- Standard classification model trained with cross-entropy loss.
- Evaluated on frequent and rare signs.

### 2. **Softmax Classification + Synthetic Dataset**
- Same architecture as above.
- Added synthetic data containing generated rare signs.

### 3. **Feature Learning (Contrastive Loss) + k-NN**
- Trains a feature extractor using contrastive loss.
- A k-Nearest Neighbors classifier is trained on extracted features.
- Variants:
  - **Train k-NN on frequent and rare signs**
  - **Train k-NN only for rare signs**

### 4. **Merged Loss Feature Learning**
- Combines contrastive loss with cross-entropy loss using a softmax head.
- Allows the feature extractor to receive supervised classification signals during training.
- Evaluated with different merge coefficients.

### 5. **Feature Learning + Random Forest for Type Classification**
- Trains a Random Forest classifier to predict whether a sign is **frequent** or **rare**.
- Uses:
  - Classifier trained on real data for frequent signs.
  - k-NN trained on synthetic rare signs for rare signs.

---

## Results

| Approach | Frequent | Rare | Total |
|----------|----------|------|-------|
| Classifier without synthetic data | 863 / 1144 (0.754) | 0 / 165 (0.000) | 863 / 1309 (0.659) |
| Classifier with synthetic data | 825 / 1144 (0.721) | 83 / 165 (0.503) | 863 / 1309 (0.693) |
| Feature learning (no CE loss, coeff=0) + KNN on rare signs | - | 101 / 165 (0.612) | - |
| Feature learning (merged loss, coeff=0.5) + KNN on rare signs | - | 97 / 165 (0.587) | - |
| Feature learning (no CE loss) + KNN + Random Forest for type classification | 863 / 1144 (0.754) | 91 / 165 (0.551) | 954 / 1309 (0.728) |

---

## Experiment Setup
- Model configurations and training parameters are stored in the `configs` and `config.py` directory.
- Dataset preprocessing scripts and synthetic data generation methods are included in the repository.
- Training and evaluation scripts are provided for each approach.

---

## Dataset
- **Name**: Russian Traffic Road Sign Dataset (RTSD)
- Contains cropped images of road signs.
- **Special split**:
  - **Frequent signs** → appear in both training and testing.
  - **Rare signs** → appear only in testing.
- Can be used to benchmark **zero-shot learning** and **few-shot learning** approaches.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ayazvaliev/Russian-Traffic-Sign-classification-methods
   cd Russian-Traffic-Sign-classification-methods
   ```
2. Install listed in `requirements.txt` dependencies (recommended to use venv):
    ```bash
    pip install -r requirements.txt
    ```
2. Download dataset and configure dataset paths, accelerator types,  in `config.py`. 
4. Select the desired experiment configuration from the `configs` directory.
5. If synthetic dataset is required it can be generated with python script:
    ```bash
    python generate_synthetic.py --output_path [OUTPUT_PATH] --config_path [CONFIG_PATH] --bg_path [BG_PATH] --icons_path [ICONS_PATH] --samples_per_class [SAMPLES_PER_CLASS] --num_workers [NUM_WORKERS]
    ```
    Further information about each argument can be found using:
    ```bash
    python generate_synthetic.py --help  
    ``` 
6. Train and evaluate selected approach:
    ```bash
    python train_and_evaluate.py --config [EXPERIMENT_CONFIG]
    ```

## Datasets used
Cropped version of [**RTSD**](https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html) has been used in demonstration purposes. Can be downloaded [here](https://drive.google.com/file/d/1fqvsd_Bn5ap0fnUhEuXS5U-mmLe5AZ1x/view?usp=sharing).

## References
1. [Faizov, Boris, et al. "Rare road signs classification."](https://computeroptics.ru/KO/PDF/KO44-2/440213.pdf)