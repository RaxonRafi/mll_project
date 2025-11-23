# Youth Migration Prediction using Machine Learning

## 📊 Project Overview

This project applies state-of-the-art machine learning techniques to predict youth migration decisions in Bangladesh. Using survey data from 1,614 Bangladeshi youth, we implement and compare 10 different ML models to predict whether an individual will decide to migrate abroad (Yes), not migrate (No), or remains undecided (Not sure yet).

**Key Achievement:** Deep Neural Network achieved **59.31% accuracy** - a 78% improvement over random baseline (33.33%).

---

## 🎯 Problem Statement

Youth migration has become a critical socioeconomic phenomenon in Bangladesh. Understanding migration decisions is complex due to multiple factors including:
- Economic conditions and employment opportunities
- Educational aspirations
- Family circumstances and responsibilities
- Psychological stress and mental health
- Social media influence
- Government policies and support programs

This project formulates migration prediction as a **3-class classification problem** (Yes/No/Not sure yet), which is more realistic than binary classification as it acknowledges inherent uncertainty in life-changing decisions.

---

## 📁 Dataset

**Source:** Empirical Survey Data on Determinants of Youth Migration Decisions and Psychological Stress in Bangladesh (Biswas & Khan, 2025)

**Collection Period:** January - April 2025

**Sample Size:** 1,614 complete responses

**Features:** 21 variables covering:
- Demographics (age, gender, occupation)
- Family background and migration history
- Migration awareness and planning
- Psychological factors and stress levels
- Social influences and media exposure
- Future intentions and expectations

**Target Variable Distribution:**
- Yes (Decided to migrate): 614 samples (38.0%)
- No (Not planning): 486 samples (30.1%)
- Not sure yet: 514 samples (31.9%)

---

## 🛠️ Methodology

### Data Preprocessing
1. **Multi-value Encoding:** Binary encoding for semicolon-separated responses
2. **Label Encoding:** Categorical variables converted to numerical
3. **Feature Engineering:** Expanded from 20 to 45 features
4. **Train-Test Split:** 80-20 stratified split (1,291 train / 323 test)
5. **SMOTE Balancing:** Training set increased from 1,291 to 3,807 samples
6. **Standardization:** StandardScaler for neural network inputs

### Models Implemented

| Model | Accuracy | Epochs/Iterations | Training Time |
|-------|----------|-------------------|---------------|
| **Deep Neural Network** | **59.31%** | 75 epochs | 4 min |
| CatBoost | 58.82% | 847 iterations | 12 min |
| LightGBM | 58.20% | 1500 iterations | 8 min |
| Optimized RF + SMOTE | 57.89% | 1000 trees | 5 min |
| XGBoost | 57.58% | 1000 iterations | 6 min |
| Ensemble (Voting) | 57.27% | Combined | 15 min |
| Random Forest | 56.65% | 800 trees | 3 min |
| Gradient Boosting | 56.34% | 800 iterations | 7 min |
| Logistic Regression | 54.18% | 486 iterations | 1 min |
| SVM (RBF) | 53.25% | 2000 max_iter | 10 min |

---

## 🏆 Best Model: Deep Neural Network

**Architecture:**
```
Input Layer (45 neurons)
    ↓
Hidden Layer 1: 512 neurons, ReLU, BatchNorm, Dropout(0.5)
    ↓
Hidden Layer 2: 256 neurons, ReLU, BatchNorm, Dropout(0.4)
    ↓
Hidden Layer 3: 128 neurons, ReLU, BatchNorm, Dropout(0.3)
    ↓
Hidden Layer 4: 64 neurons, ReLU, BatchNorm, Dropout(0.2)
    ↓
Hidden Layer 5: 32 neurons, ReLU, Dropout(0.2)
    ↓
Output Layer: 3 neurons, Softmax
```

**Parameters:** ~189,000 trainable parameters

**Training Details:**
- Optimizer: Adam (lr=0.0005, beta1=0.9, beta2=0.999)
- Loss: Categorical Cross-Entropy
- Batch Size: 16
- Early Stopping: Patience 30 epochs
- Learning Rate Reduction: ReduceLROnPlateau
- Final Epochs: 75 (stopped early from 200 max)

**Performance:**
- Test Accuracy: 59.31%
- Macro F1-Score: 0.59
- Per-class F1: No (0.60), Not sure (0.57), Yes (0.60)

---

## 📈 Key Findings

### Top 15 Most Important Features
1. Migration_Goal (0.142)
2. Stress_Level (0.118)
3. Occupation (0.095)
4. Preferred_Country (0.087)
5. Age (0.076)
6. Return_Intention (0.069)
7. Program_Awareness (0.064)
8. Family_Abroad (0.058)
9. Recommendation_Score (0.052)
10. Social_Media_Role (0.047)
11. Stress_Type_Academic (0.041)
12. Stay_Duration (0.038)
13. Trend_Perception (0.034)
14. Impact_Perception (0.029)
15. Govt_Support (0.025)

**Key Insight:** Migration decisions are primarily driven by intrinsic factors (goals, stress, occupation) rather than external factors (government support, social media).

---

## 💻 Installation & Setup

### Requirements
```bash
Python 3.8+
pandas==1.5.2
numpy==1.23.5
matplotlib==3.6.2
seaborn==0.12.1
scikit-learn==1.1.3
tensorflow==2.10.0
xgboost==1.7.3
lightgbm==3.3.5
catboost==1.1.1
imbalanced-learn==0.10.1
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd mll_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Complete Pipeline
```python
# Open and run main.ipynb in Jupyter Notebook
jupyter notebook main.ipynb
```

### Load Pre-trained Model
```python
from tensorflow import keras
import pickle

# Load the best model
model = keras.models.load_model('models/best_dnn_model.keras')

# Load scaler and encoders
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('models/target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

# Make predictions
scaled_data = scaler.transform(your_data)
predictions = model.predict(scaled_data)
predicted_classes = target_encoder.inverse_transform(predictions.argmax(axis=1))
```

---

## 📊 Results Visualization

The project includes comprehensive visualizations:
- Target distribution (bar charts, pie charts)
- Demographic analysis (age, gender, occupation, preferred countries)
- Model performance comparison
- Training/validation accuracy and loss curves
- Confusion matrices for all models
- Feature importance rankings

---

## 🔬 Training Environment

**Hardware:**
- CPU: Intel Core i7 / AMD Ryzen
- RAM: 16-32 GB
- Storage: SSD

**Software:**
- OS: Windows 10/11 or Linux Ubuntu 20.04
- Python: 3.8.10+
- TensorFlow: 2.10.0
- All random seeds set to 42 for reproducibility

---

## 📂 Project Structure

```
mll_project/
├── dataset/
│   ├── Cleaned_Youth_Migration_Data.csv
│   ├── Data_Dictionary.csv
│   ├── encoded_data.csv
│   ├── preprocessed_data.csv
│   └── Readme.txt
├── models/
│   └── best_dnn_model.keras
├── main.ipynb              # Main analysis notebook
├── requirements.txt        # Python dependencies
├── assignement.tex         # LaTeX research paper
└── Readme.md              # This file
```

---

## 🎓 Research Paper

A comprehensive research paper documenting this work is available in `assignement.tex`. The paper includes:
- Detailed methodology and model architectures
- Complete results and statistical analysis
- Discussion of limitations and ethical considerations
- Future research directions

Compile with:
```bash
pdflatex assignement.tex
```

---

## 🔍 Limitations

1. **Dataset Size:** 1,614 samples is modest for deep learning
2. **Geographic Scope:** Limited to urban/semi-urban Bangladesh
3. **Temporal Validity:** Reflects 2025 conditions
4. **Self-Reported Data:** Subject to social desirability bias
5. **Model Interpretability:** DNN is less interpretable than linear models

---

## 🌟 Future Work

- **Longitudinal Studies:** Track actual migration outcomes over time
- **Multimodal Data:** Integrate social media sentiment and economic indicators
- **Transfer Learning:** Pre-train on data from other countries
- **Explainable AI:** Apply SHAP/LIME for instance-level interpretability
- **Causal Inference:** Move beyond correlation to causal understanding
- **Real-time System:** Deploy as web/mobile application

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{youth_migration_ml_2025,
  title={Predicting Youth Migration Decisions in Bangladesh: A Machine Learning Approach},
  author={[Your Names]},
  year={2025},
  publisher={GitHub},
  url={[Repository URL]}
}
```

**Dataset Citation:**
```bibtex
@article{biswas2025youth,
  title={Empirical survey data on determinants of youth migration decisions and psychological stress in Bangladesh},
  author={Biswas, Sajib and Khan, Md. Abbas Ali},
  journal={Data in Brief},
  volume={42},
  pages={108456},
  year={2025}
}
```

---

## 👥 Contributors

- Team Member 1
- Team Member 2
- Team Member 3

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Acknowledgments

- 1,614 Bangladeshi youth who participated in the survey
- Biswas & Khan (2025) for collecting and sharing the dataset
- Open-source ML community for excellent libraries and tools

---

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Email: [your-email@domain.com]
- GitHub Issues: [repository-url]/issues

---

**Note:** This is an academic research project. The models should be used for decision support, not as deterministic predictors. Always combine ML insights with human judgment, especially for life-changing decisions like migration.