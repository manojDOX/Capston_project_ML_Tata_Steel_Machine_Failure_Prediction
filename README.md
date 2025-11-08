# Tata Steel: Predictive Maintenance for Machine Failure

## Project Overview

This machine learning project develops a robust predictive model for TATA Steel to forecast machinery failures before they occur, enabling a strategic shift from costly reactive maintenance to efficient proactive maintenance.

### Business Problem

Unpredictable machine breakdowns cause major financial losses and operational disruptions in steel manufacturing. Without data-driven forecasting, maintenance occurs only after failures, leading to:
- Unplanned downtime
- Increased emergency repair costs
- Reduced production output

This project leverages historical operational data to build a classification model that anticipates failures, allowing TATA Steel to schedule repairs preemptively and optimize overall equipment effectiveness (OEE).

## Dataset

The dataset contains real-time operational parameters from TATA Steel's machinery. It was exceptionally clean with no missing or duplicate values.

### Features

- **Product Type**: Quality of product being manufactured (Low, Medium, High)
- **Air temperature [K]**: Ambient temperature around the machine
- **Process temperature [K]**: Internal temperature during operation
- **Rotational speed [rpm]**: Tool rotation speed
- **Torque [Nm]**: Rotational force applied
- **Tool wear [min]**: Cumulative tool usage time
- **Failure Flags**: Binary indicators for specific failure modes
  - Heat Dissipation Failure (HDF)
  - Power Failure (PWF)
  - Overstrain Failure (OSF)
  - Tool Wear Failure (TWF)

## Key Insights from EDA

### Class Imbalance
- Machine failures constitute only **1.6%** of all recorded operations
- Severe imbalance required specialized handling techniques

### Failure Patterns
- Majority of failures occur during **'Low' quality product** production
- Most common failure types: **Heat Dissipation Failure (HDF)** and **Overstrain Failure (OSF)**
- Strong negative correlation between Rotational Speed and Torque

### Critical Operating Zones
Highest failure rates occur in:
- **High-torque/low-speed** operations
- **Low-torque/high-speed** operations
- Safest range: moderate speed and moderate torque

## Hypothesis Testing

All tests conducted at 5% significance level:

1. **Process Temperature vs Failures**
   - Test: Welch's Independent T-test
   - Result: Process temperature is significantly higher for failed machines ✓

2. **Torque vs Rotational Speed**
   - Test: Pearson Correlation Test
   - Result: Strong negative correlation confirmed ✓

3. **Product Type vs Failure Rate**
   - Test: Chi-squared (χ²) test
   - Result: Significant association between product type and failure likelihood ✓

## Feature Engineering

### Pre-processing Steps

1. **Column Standardization**: Removed special characters and spaces from column names
2. **Categorical Encoding**: Applied Ordinal Encoding to Product Type (L=1, M=2, H=3)
3. **Multicollinearity Handling**: 
   - Conducted VIF analysis
   - Removed Process_temperature_K_ due to high multicollinearity with Air_temperature_K_
4. **Outlier Management**: Retained outliers as they represent critical failure conditions
5. **Data Splitting**: 70-30 train-test split with stratification to preserve class balance

## Model Development

### Models Evaluated

Three tree-based models were trained:
- **LightGBM**
- **XGBoost**
- **Random Forest**

### Imbalance Handling

Used built-in class weight parameters (`class_weight` and `scale_pos_weight`) to penalize misclassifications of the minority class, proving more effective than SMOTE for this dataset.

## Model Performance

| Model | Recall | Precision | F1-Score | Key Characteristics |
|-------|--------|-----------|----------|---------------------|
| **LightGBM** | 0.84 | 0.41 | 0.55 | Good failure detection, many false alarms |
| **XGBoost** | 0.82 | 0.61 | 0.70 | Well-balanced performance |
| **Random Forest** | 0.77 | **0.95** | **0.85** | **Most reliable predictions** |

### Recommended Model

**Random Forest (Baseline)** is the recommended solution:
- **95% precision**: When it predicts a failure, it's correct 95% of the time
- **77% recall**: Catches 77% of all actual failures
- High reliability prevents alert fatigue and builds trust with maintenance teams
- Extremely accurate failure predictions enable confident maintenance scheduling

  <img width="846" height="624" alt="image" src="https://github.com/user-attachments/assets/19257029-ae4c-4423-9af7-92cfa7bfb7f1" />


## Business Impact

Deploying this predictive model enables TATA Steel to achieve:

### Operational Benefits
- **Reduced Unplanned Downtime**: Proactive maintenance keeps production lines running consistently
- **Lower Maintenance Costs**: 
  - Eliminates emergency repair premiums
  - Reduces labor overtime
  - Prevents secondary equipment damage
  - Minimizes rush-ordered parts
- **Improved Production Efficiency**: Greater machine reliability leads to predictable output and enhanced operational effectiveness

### Strategic Value
- Transition from reactive to proactive maintenance strategy
- Data-driven decision making for maintenance scheduling
- Optimized Overall Equipment Effectiveness (OEE)
- Enhanced production planning capabilities

## Technologies Used

- Python
- Machine Learning: LightGBM, XGBoost, Random Forest
- Statistical Analysis: Hypothesis testing, VIF analysis
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn

## Conclusion

This project successfully developed a highly accurate machine learning model capable of predicting machinery failures from operational data. The Random Forest model provides a practical, reliable tool for TATA Steel's maintenance teams, enabling significant cost savings and operational improvements through predictive maintenance strategies.
