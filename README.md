# MovieLens Recommender System

A comprehensive movie recommendation system built with scikit-learn, demonstrating multiple recommendation approaches and advanced machine learning techniques on the MovieLens dataset.

## 🎯 Project Overview

This project implements and compares various recommendation algorithms using the MovieLens dataset, showcasing advanced scikit-learn techniques including custom transformers, pipeline optimization, and hybrid modeling approaches. The system provides both rating prediction and top-N movie recommendations.

## 🚀 Features

### Recommendation Approaches
- **Collaborative Filtering**: User-based and item-based similarity models
- **Matrix Factorization**: SVD and NMF for dimensionality reduction
- **Content-Based Filtering**: Genre and movie metadata analysis
- **Hybrid Models**: Combining multiple approaches for improved performance
- **Deep Feature Engineering**: User profiles, temporal patterns, and interaction features

### Advanced ML Techniques
- Custom scikit-learn transformers and pipelines
- Sparse matrix optimization for large-scale data
- Cross-validation with temporal splits (avoiding data leakage)
- Hyperparameter tuning with GridSearchCV
- Custom evaluation metrics for recommendation quality

## 📊 Dataset

**MovieLens 1M Dataset**
- 1 million ratings from 6,000 users on 4,000 movies
- User demographics (age, gender, occupation)
- Movie metadata (genres, release year)
- Ratings scale: 1-5 stars

## 🛠️ Technical Implementation

### Core Technologies
- **Python 3.8+**
- **scikit-learn**: Primary ML framework
- **NumPy & Pandas**: Data manipulation
- **SciPy**: Sparse matrix operations
- **Matplotlib & Seaborn**: Visualization

### Key Components

#### Data Processing Pipeline
```python
# Custom transformer for user feature engineering
class UserProfileTransformer(BaseEstimator, TransformerMixin)

# Movie content feature extraction
class MovieContentTransformer(BaseEstimator, TransformerMixin)

# Sparse matrix handler for user-item interactions
class SparseMatrixTransformer(BaseEstimator, TransformerMixin)
```

#### Model Architecture
- **Baseline Models**: Global averages, user/movie biases
- **Similarity Models**: Cosine, Pearson correlation
- **Matrix Factorization**: TruncatedSVD, NMF with regularization
- **Ensemble Methods**: Weighted combinations of multiple models

## 📈 Evaluation Metrics

### Rating Prediction
- **RMSE** & **MAE**: Standard regression metrics
- **R²**: Explained variance in ratings

### Recommendation Quality
- **Precision@K** & **Recall@K**: Top-N recommendation accuracy
- **NDCG**: Normalized discounted cumulative gain
- **Coverage**: Catalog coverage percentage
- **Diversity**: Intra-list diversity using genre distribution

## 🏗️ Project Structure

```
movielens-recommender/
├── data/
│   ├── raw/                    # Original MovieLens files
│   └── processed/              # Cleaned and engineered features
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── feature_engineering.py # Custom transformers
│   ├── models.py              # Recommendation algorithms
│   ├── evaluation.py          # Metrics and validation
│   └── utils.py               # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_results_analysis.ipynb
├── config/
│   └── model_configs.yaml     # Hyperparameter configurations
├── results/
│   ├── model_performance.csv  # Benchmarking results
│   └── visualizations/        # Performance plots
└── requirements.txt
```

## 🚦 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/movielens-recommender
cd movielens-recommender
pip install -r requirements.txt
```

### Quick Start
```python
from src.models import HybridRecommender
from src.data_processing import load_movielens_data

# Load data
ratings, movies, users = load_movielens_data()

# Initialize hybrid model
recommender = HybridRecommender(
    collab_weight=0.7,
    content_weight=0.3,
    n_factors=50
)

# Train model
recommender.fit(ratings, movies, users)

# Get recommendations for user
recommendations = recommender.recommend(user_id=1, n_items=10)
```

## 📊 Results

### Model Performance Comparison
| Model | RMSE | Precision@10 | Recall@10 | Coverage |
|-------|------|-------------|-----------|----------|
| Global Average | 1.126 | - | - | - |
| User-Based CF | 0.943 | 0.156 | 0.087 | 0.68 |
| Item-Based CF | 0.951 | 0.162 | 0.091 | 0.72 |
| SVD (50 factors) | 0.924 | 0.171 | 0.095 | 0.89 |
| **Hybrid Model** | **0.912** | **0.184** | **0.108** | **0.91** |

### Key Insights
- Matrix factorization methods outperform neighborhood-based approaches
- Hybrid models achieve best overall performance across all metrics
- Content-based features improve cold-start recommendation quality
- Temporal patterns enhance rating prediction accuracy

## 🔬 Advanced Features

### Custom Scikit-learn Components
- **TimeAwareSplitter**: Custom CV splitter respecting temporal order
- **RecommenderEvaluator**: Comprehensive evaluation suite
- **FeatureUnion**: Combining sparse and dense features efficiently

### Scalability Optimizations
- Sparse matrix operations for memory efficiency
- Incremental learning for large datasets
- Batch prediction for real-time serving

## 📚 Research Applications

This implementation demonstrates several advanced ML concepts:
- **Cold Start Problem**: Handling new users and items
- **Implicit Feedback**: Learning from rating patterns
- **Temporal Dynamics**: Time-aware recommendation models
- **Fairness**: Analyzing recommendation bias across user groups

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Deep learning integration (neural collaborative filtering)
- Real-time recommendation serving
- A/B testing framework
- Additional evaluation metrics

## 📄 License

MIT License - see LICENSE file for details.

## 📖 References

- Koren, Y. (2009). Matrix factorization techniques for recommender systems
- Ricci, F. et al. (2015). Recommender Systems Handbook
- MovieLens Dataset: https://grouplens.org/datasets/movielens/

---

**Built with ❤️ and scikit-learn**