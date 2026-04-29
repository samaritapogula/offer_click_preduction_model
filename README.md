```markdown
# Offer Click Prediction Model

Binary classification model predicting customer click probability on personalized offers using LightGBM with engineered CTR features and temporal context.

## Problem

Predict whether a customer will click on an offer impression given customer-offer interaction history and merchant metadata. Dataset: 770K impressions with 372 anonymised features.

## Solution

**Data Pipeline**
- Reduced memory footprint from 13 GB to 685 MB (95% reduction) through dtype optimisation
- Cleaned and deduplicated 3 datasets: transactions, events, offer metadata
- Standardised timestamps and categorical encodings

**Feature Engineering**
- CTR velocity: 1d, 3d, 7d, 14d, 30d ratios and trends
- Click-impression engagement ratios across time windows
- Merchant and industry-level CTR aggregations
- Temporal features: hour-of-day, day-of-week buckets
- Removed >90% missing, constant, and highly correlated features (r > 0.95)

**Model**
- LightGBM with 5-fold stratified cross-validation
- L1/L2 regularisation, feature subsampling, early stopping
- **Mean validation AUC: 0.9558** (train 0.9910)
- Train-validation gap: <3.5%

## Results

| Fold | Train AUC | Val AUC | Best Iter |
|------|-----------|---------|-----------|
| 1 | 0.9917 | 0.9567 | 180 |
| 2 | 0.9910 | 0.9549 | 168 |
| 3 | 0.9917 | 0.9572 | 181 |
| 4 | 0.9904 | 0.9568 | 160 |
| 5 | 0.9900 | 0.9534 | 149 |
| **Avg** | **0.9910** | **0.9558** | **168** |



## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
lightgbm>=4.0.0
scikit-learn>=1.0.0
```



## Key Insights

- 38% feature reduction improved stability without AUC loss
- CTR trends over 30 days more predictive than raw values
- Merchant aggregations capture cross-offer behaviour
- Early stopping consistently at iteration 168

---
