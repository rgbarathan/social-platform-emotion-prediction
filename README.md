# 🧠 Social Platform Emotion Prediction

A machine learning project that predicts user emotions based on social media platform usage patterns and behavior analytics.

## 📋 Project Overview

This project analyzes social media user behavior to predict dominant emotions using a Random Forest classifier. The model achieves **98% accuracy** and **99.8% ROC-AUC score** in classifying emotions from user engagement patterns.

### 🎯 Key Features

- **Emotion Classification**: Predicts 7 different emotional states
- **High Accuracy**: 98% classification accuracy
- **Outstanding Performance**: 99.8% ROC-AUC score
- **Clean Datasets**: Pre-processed, production-ready data
- **Feature Analysis**: Interactive visualization of feature importance
- **Professional Pipeline**: Optimized code with error handling

## 🏗️ Project Structure

```
Project Social Platform/
├── Social Platform.py      # Main analysis script (simplified)
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── dataset/
    ├── train.csv          # Clean training data (924 samples)
    ├── val.csv            # Clean validation data (132 samples)
    ├── test.csv           # Clean test data (101 samples)
    └── original_backup/   # Backup of original corrupted files from Kaggle
        ├── train_original.csv
        ├── val_original.csv
        └── test_original.csv
```

**Note**: The `original_backup/` folder contains the unprocessed Kaggle data for transparency and reproducibility. The main dataset files contain the cleaned, production-ready data used for analysis.

## 📊 Dataset Information

### 📄 **Data Source & Attribution**
- **Original Source**: Kaggle - Social Media Usage and Emotional Wellbeing Dataset
- **Original Size**: ~1,249 records across multiple social media platforms
- **Collection Method**: Survey-based data collection from social media users aged 21-35

### 🧹 **Data Quality & Preprocessing**
**Important Note**: The original Kaggle dataset contained significant data quality issues that required extensive preprocessing:

#### **🚨 Issues Identified in Original Dataset:**
- **Corrupted entries**: Mixed data types in numeric columns (text in age fields)
- **Missing values**: Incomplete records across multiple features
- **Invalid categories**: Inconsistent emotion and platform labels
- **Data contamination**: Foreign language text mixed with English data
- **Outliers**: Unrealistic values in behavioral metrics
- **Encoding errors**: Character encoding issues affecting data integrity

#### **✅ Data Cleaning Process Applied:**
1. **Data Validation**: Removed entries with invalid data types
2. **Missing Data Handling**: Eliminated incomplete records (maintaining data integrity over quantity)
3. **Range Validation**: Applied realistic bounds to behavioral metrics:
   - Age: 21-35 years (active social media demographic)
   - Daily usage: 40-200 minutes (realistic usage patterns)
   - Engagement metrics: Validated against platform norms
4. **Category Standardization**: Ensured consistent emotion and platform labels
5. **Encoding Correction**: Fixed character encoding and text formatting issues
6. **Quality Assurance**: Final validation pass to ensure data consistency

#### **📈 Data Retention:**
- **Original Dataset**: 1,249 samples
- **After Cleaning**: 1,157 samples (92.6% retention rate)
- **Quality Improvement**: 100% clean, validated data vs. corrupted original

### 📁 **Dataset Structure (Post-Cleaning)**

### Input Features:
- **Age**: User age (21-35 years, cleaned range)
- **Gender**: Male, Female, Non-binary
- **Platform**: Social media platform used (7 platforms)
  - Instagram, Twitter, Facebook, LinkedIn, WhatsApp, Telegram, Snapchat
- **Daily_Usage_Time_minutes**: Minutes spent per day (1-1440)
- **Posts_Per_Day**: Number of posts created daily (0-50)
- **Likes_Received_Per_Day**: Daily likes received (0-10,000)
- **Comments_Received_Per_Day**: Daily comments received (0-1,000)
- **Messages_Sent_Per_Day**: Daily private messages sent (0-500)

### Target Variable:
- **Dominant_Emotion**: 7 categories
  - Agression
  - Anger
  - Anxiety
  - Boredom
  - Happiness
  - Neutral
  - Sadness

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Project Social Platform"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

Execute the main script:
```bash
python "Social Platform.py"
```

## 📈 Model Performance

### Results Summary:
- **Overall Accuracy**: 98%
- **ROC-AUC Score**: 99.8%
- **Precision**: 93-100% across all classes
- **Recall**: 94-100% across all classes
- **F1-Score**: 96-100% across all classes

### Performance by Emotion:

| Emotion    | Precision | Recall | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Agression  | 100%      | 100%    | 100%     | 9       |
| Anger      | 100%      | 95%     | 98%      | 22      |
| Anxiety    | 100%      | 94%     | 97%      | 16      |
| Boredom    | 93%       | 100%    | 96%      | 13      |
| Happiness  | 100%      | 100%    | 100%     | 27      |
| Neutral    | 93%       | 100%    | 97%      | 14      |
| Sadness    | -         | -       | -        | 0       |

### 🔝 Top Predictive Features:
1. **Daily_Usage_Time_minutes** (18.3%) - Most important predictor
2. **Likes_Received_Per_Day** (17.1%) - Social validation metric  
3. **Age** (16.4%) - Demographic factor
4. **Comments_Received_Per_Day** (14.2%) - Engagement depth
5. **Messages_Sent_Per_Day** (12.5%) - Communication activity

## 🛠️ Technical Implementation

### Machine Learning Pipeline:

1. **Clean Data Loading**
   - Pre-processed datasets with quality validation
   - Consistent column naming and data types
   - Realistic value ranges enforced

2. **Feature Engineering**
   - Label encoding for categorical variables
   - Proper train/validation/test splits maintained
   - Feature importance analysis

3. **Model Training**
   - Random Forest Classifier (100 estimators)
   - Cross-validation on training data
   - Robust prediction pipeline

4. **Evaluation & Visualization**
   - Comprehensive classification metrics
   - ROC-AUC analysis for multi-class classification
   - Interactive feature importance plots

### Key Libraries Used:
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **plotly**: Interactive data visualization
- **numpy**: Numerical computations

## 📊 Output

The script generates:

1. **Dataset Loading Summary**: Clean data confirmation
2. **Encoding Information**: Categorical variable mappings
3. **Model Training Status**: Training completion confirmation
4. **Performance Metrics**: Detailed classification report
5. **ROC-AUC Score**: Overall discrimination capability
6. **Feature Importance Plot**: Interactive visualization
7. **Top Features Ranking**: Most predictive behaviors
8. **Emotion Label Mapping**: Class encoding reference

## 🎯 Applications

### Business Use Cases:
- **Mental Health Monitoring**: Early detection of anxiety/depression patterns
- **Content Personalization**: Emotion-aware content recommendation
- **User Experience**: Adaptive interfaces based on emotional state
- **Community Management**: Proactive intervention for negative emotions
- **Marketing Strategy**: Emotion-based campaign targeting
- **Platform Optimization**: Feature development guided by emotional impact

### Research Applications:
- Social media psychology research
- Digital wellbeing studies
- Behavioral pattern analysis
- Emotion regulation research
- Human-computer interaction studies

## 🔧 Data Quality & Preprocessing

### Data Cleaning Accomplished:
- **Corrupted Data Removal**: Eliminated mixed data types and invalid entries
- **Outlier Filtering**: Applied realistic bounds to all numeric features
- **Missing Data Handling**: Removed incomplete records
- **Categorical Validation**: Ensured valid categories for all categorical features
- **Age Range Filtering**: Limited to realistic social media user ages (21-35)

### Quality Metrics:
- **Data Retention**: 92.6% of original data retained after cleaning
- **Consistency**: 100% consistent data types and formats
- **Completeness**: No missing values in final datasets
- **Validity**: All values within realistic ranges

## 🔬 Model Validation

### Robust Evaluation:
- **Independent Test Set**: Never seen during training
- **Stratified Splits**: Maintained class distributions
- **Multiple Metrics**: Precision, recall, F1-score, ROC-AUC
- **Feature Importance**: Interpretable model insights

### Performance Consistency:
- **High Precision**: Minimal false positive predictions
- **High Recall**: Excellent coverage of true cases
- **Balanced F1**: Optimal precision-recall trade-off
- **Outstanding ROC-AUC**: Superior class discrimination

## 📝 Notes

- **Data Source**: Original dataset sourced from Kaggle with extensive quality issues requiring comprehensive preprocessing
- **Data Cleaning**: Applied rigorous data validation and cleaning processes to ensure research-grade data quality
- **Data Privacy**: All user data is anonymized and used for research purposes only  
- **Model Validation**: Excellent performance achieved through proper data preprocessing and validation methodology
- **Reproducibility**: Original corrupted data preserved in `original_backup/` for transparency and research integrity
- **Data Quality**: Current dataset represents industry-standard clean data practices and validation procedures
- **Academic Integrity**: Full attribution to original Kaggle source with transparent documentation of all preprocessing steps

## 🔄 Future Enhancements

### Potential Improvements:
- **Real-time Prediction**: API development for live emotion detection
- **Temporal Analysis**: Time-series emotion tracking
- **Deep Learning**: Neural network comparison studies
- **Multi-modal Data**: Integration of text/image sentiment analysis
- **Larger Datasets**: Scaling to thousands of users

### Research Extensions:
- **Longitudinal Studies**: Long-term emotion pattern tracking
- **Cultural Analysis**: Cross-cultural emotion expression differences
- **Intervention Studies**: Effectiveness of emotion-based interventions
- **Platform Comparison**: Emotion patterns across different social media

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:
- Additional feature engineering
- Model optimization techniques
- Visualization enhancements
- Documentation improvements
- Dataset expansion

## � Data Attribution & Citation

### **Original Dataset Source**
- **Platform**: Kaggle
- **Dataset**: Social Media Usage and Emotional Wellbeing 
- **Original Contributors**: Various data contributors on Kaggle platform
- **License**: Public domain / Open source (as per Kaggle terms)

### **Data Processing Citation**
If using this cleaned dataset for research or academic purposes, please cite:
```
Social Platform Emotion Prediction Dataset (Cleaned Version)
Processed from: Kaggle Social Media Usage and Emotional Wellbeing Dataset
Data Cleaning and Validation: [Author: rgbarathan]
GitHub Repository: https://github.com/rgbarathan/social-platform-emotion-prediction
Date: November 2025
```

### **Acknowledgments**
- Original dataset contributors on Kaggle platform
- Drexel University for academic framework and guidance
- Open-source community for machine learning tools and libraries

## �📄 License

This project is for educational purposes. Please ensure appropriate data usage rights when applying to real datasets.

## 👨‍💻 Author

Created for Drexel University Assignment 3 - Social Platform Analysis

---

*Last Updated: November 2, 2025*
*Final Version: Production-Ready AI System with Clean Datasets*