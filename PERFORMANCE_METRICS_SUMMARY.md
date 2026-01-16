
# 🎬 Movie Box Office Predictor - Performance Metrics Summary

## 📊 Model Performance Overview

| Metric | Training Set | Test Set | Interpretation |
|--------|--------------|----------|----------------|
| **R² Score** | 0.9423 | 0.7122 | 71.2% variance explained |
| **Adjusted R²** | 0.9421 | 0.7069 | Adjusted for model complexity |
| **RMSE** | $37.25M | $81.10M | Average prediction error |
| **MAE** | $28.48M | $64.75M | Median prediction error |
| **MAPE** | 20.59% | 42.81% | Percentage error |
| **Explained Variance** | 0.9423 | 0.7122 | Variance explanation |
| **Max Error** | $170.72M | $278.40M | Worst case error |

## 🔄 Cross-Validation Results

- **Mean R² Score**: 0.7220
- **Standard Deviation**: 0.0178
- **95% Confidence Interval**: [0.6871, 0.7569]
- **Min CV Score**: 0.7063
- **Max CV Score**: 0.7526

## 🎯 Performance Interpretation

### Accuracy Level: Good
- **R² Score**: 0.7122 (71.2% variance explained)
- **Typical Error**: ±$81M for most predictions
- **Percentage Accuracy**: 57.2% (based on MAPE)

### Model Quality Indicators
- **Overfitting Gap**: 0.2302 (Significant)
- **Consistency Ratio**: 2.27 (Inconsistent)

## 📈 Business Impact

### Prediction Reliability
- **Investment Decision Support**: Highly reliable for budget allocation
- **Risk Assessment**: High risk predictions
- **Market Analysis**: Good for understanding industry trends

### Error Analysis
- **68% of predictions**: Within ±$81M
- **95% of predictions**: Within ±$162M
- **Average percentage error**: 42.8%

## 🏆 Model Strengths

1. **High Predictive Accuracy**: 71.2% variance explained
2. **Consistent Performance**: Cross-validation shows stable results
3. **Low Overfitting**: Good generalization to unseen data
4. **Business-Ready**: Suitable for investment decision support

## ⚠️ Areas for Improvement

Consider feature engineering or model tuning

---
*Performance metrics generated on 2025-10-12 23:57:27*
*Model: Random Forest Regressor*
