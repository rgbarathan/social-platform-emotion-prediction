# 🧠 Social Platform Emotion Prediction

A machine learning project that predicts user emotions based on social media platform usage patterns and behavior analytics.

## 📋 Project Overview

This project analyzes social media user behavior to predict dominant emotions using a Random Forest classifier. The model achieves **98% accuracy** and **99.8% ROC-AUC score** in classifying emotions from user engagement patterns.

### 🎯 Key Features

- **Advanced Model Comparison**: Tests 4+ algorithms automatically
- **Dynamic User Generation**: Creates new random users each run 🆕
- **Emotion Classification**: Predicts 7 different emotional states  
- **High Accuracy**: 98% classification accuracy with Random Forest
- **Outstanding Performance**: 99.8% ROC-AUC score
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Auto Model Selection**: Automatically selects best performer
- **Unknown User Prediction**: Predict emotions for new users
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
- **Original Source**: [Social Media Usage and Emotional Well-being Dataset](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being) - Kaggle
- **Author**: emirhanai
- **Platform**: Kaggle Datasets
- **Original Size**: ~1,249 records across multiple social media platforms
- **Collection Method**: Survey-based data collection from social media users aged 21-35
- **License**: Public Dataset (Kaggle)

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

Execute the main script for the complete experience:
```bash
python "Social Platform.py"
```

**🎮 Interactive Experience**: After the analysis completes, you'll be prompted to explore additional features:
- Enter `y` for an interactive menu with 8 different demo options
- Press Enter to finish and see just the core analysis

### 🎮 **Integrated Demo System**

The main script now includes a comprehensive interactive menu:

1. **🎲 Generate More Random Users** - Create fresh user profiles
2. **🎯 Create Extreme User Scenarios** - Test edge cases  
3. **🌐 Platform-Specific Showcase** - Focus on individual platforms
4. **⚡ Quick Single User Prediction** - Instant random prediction
5. **🔄 Compare Multiple Generations** - Side-by-side user comparisons
6. **📊 Model Performance Deep Dive** - Detailed algorithm analysis
7. **🚀 Interactive Prediction Mode** - Manual user input system
8. **❌ Exit** - Return to terminal

### 🔬 **Advanced Model Comparison Suite**

**NEW FEATURE**: Comprehensive algorithm comparison and automatic best model selection!

#### **What's Included**
- **4+ Machine Learning Algorithms**: Random Forest, Gradient Boosting, SVM, Logistic Regression
- **Advanced Models**: XGBoost, LightGBM (when available)
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Automatic Selection**: System picks the best performing model
- **Visual Comparison**: Interactive charts comparing all models

#### **Algorithm Performance Results**
| Algorithm | Accuracy | F1-Score | ROC-AUC | CV Score |
|-----------|----------|----------|---------|----------|
| 🥇 Random Forest | 98.02% | 0.9803 | 0.9982 | 98.73% ± 0.24% |
| 🥈 Gradient Boosting | 97.03% | 0.9704 | 0.9926 | 98.34% ± 0.39% |
| 🥉 Support Vector Machine | 74.26% | 0.7444 | 0.9508 | 74.83% ± 1.62% |
| 🏅 Logistic Regression | 53.47% | 0.5221 | 0.8623 | 54.54% ± 0.84% |

#### **Key Insights**
- **Random Forest** wins with outstanding 98% accuracy
- **Tree-based models** significantly outperform linear models
- **Cross-validation** confirms consistent performance
- **Automatic selection** ensures optimal results

### 🔮 Unknown User Prediction

**NEW FEATURE**: Predict emotions for new users not in the training dataset!

#### **🎲 Dynamic User Generation** 🆕
The system now generates **completely new random users every time** you run it:

- **8 Platform Types**: Instagram, Facebook, Twitter, LinkedIn, Snapchat, TikTok, WhatsApp, Telegram
- **Realistic Patterns**: Age-appropriate demographics and platform-specific behaviors
- **Diverse Scenarios**: From social media addicts to digital minimalists
- **Fresh Each Run**: Never see the same demo users twice!

```bash
# Every run shows different random users
python "Social Platform.py"
```

**Example Dynamic Output:**
```
🎲 New random users created each run!

👤 Random User #1: Content Creator (Instagram)
   📊 Profile: 24yr Female on Instagram  
   ⏱️ Usage: 180 min/day, 4 posts/day
   💬 Engagement: 95 likes, 12 comments, 35 messages
   🎯 Predicted Emotion: Happiness (Confidence: 67.2%)
```

#### **📱 Platform-Specific User Types**
- **Instagram**: Influencer, Content Creator, Photo Enthusiast, Story Addict
- **LinkedIn**: Professional, Job Seeker, Industry Expert, Networker  
- **TikTok**: Creator, Viral Chaser, Dance Enthusiast, Comedy Fan
- **Facebook**: Family User, News Reader, Community Member, Casual Browser
- **And more**: Twitter activists, Snapchat story posters, WhatsApp chatters

#### **Quick Prediction Function**
```python
# After running the main script, use:
emotion, confidence = predict_user_emotion(
    age=25,
    gender='Female',  # 'Male' or 'Female'
    platform='Instagram',  # Available: Facebook, Instagram, Twitter, LinkedIn, Snapchat, Whatsapp, Telegram
    daily_usage_time=120,  # Minutes per day
    posts_per_day=2,
    likes_received=30,
    comments_received=8,
    messages_sent=25
)
print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.1f}%)")
```

#### **Interactive Prediction System**
```python
# Run interactive mode for multiple predictions
interactive_prediction()
```

#### **Example Usage**
The main script includes everything:
```bash
# Complete system with interactive options
python "Social Platform.py"

# Optional: Individual example files (legacy)
python predict_examples.py     # Static reference examples  
python dynamic_demo.py         # Dynamic showcase (now integrated)
```

**Recommended**: Use the main script's interactive menu for the best experience!

#### **Supported Parameters**
- **Age**: User's age (numeric, 13-65)
- **Gender**: 'Male' or 'Female'  
- **Platform**: Facebook, Instagram, Twitter, LinkedIn, Snapchat, Whatsapp, Telegram
- **Daily Usage Time**: Minutes spent daily (1-1440)
- **Posts Per Day**: Average posts published (0-50)
- **Likes Received**: Average daily likes (0-10,000)
- **Comments Received**: Average daily comments (0-1,000)
- **Messages Sent**: Daily messages sent (0-500)

#### **Confidence Interpretation**
- 🟢 **80-100%**: Very High Confidence
- 🟡 **60-79%**: High Confidence  
- 🟠 **40-59%**: Moderate Confidence
- 🔴 **<40%**: Low Confidence (unusual pattern)

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

### Advanced Machine Learning Pipeline:

1. **Clean Data Loading**
   - Pre-processed datasets with quality validation
   - Consistent column naming and data types
   - Realistic value ranges enforced

2. **Model Comparison Framework** 🆕
   - **Multiple Algorithms**: Random Forest, Gradient Boosting, SVM, Logistic Regression
   - **Advanced Models**: XGBoost, LightGBM (optional)
   - **Cross-Validation**: 5-fold stratified CV for each model
   - **Automatic Selection**: Best model chosen by accuracy
   - **Performance Metrics**: Comprehensive evaluation across all models

3. **Feature Engineering**
   - Label encoding for categorical variables
   - Feature scaling for appropriate algorithms
   - Proper train/validation/test splits maintained
   - Feature importance analysis for winner model

4. **Model Training & Selection**
   - Parallel model training and evaluation
   - Statistical comparison with cross-validation
   - Automatic best model selection
   - Robust prediction pipeline with the winner

5. **Evaluation & Visualization**
   - Comprehensive classification metrics
   - ROC-AUC analysis for multi-class classification
   - Interactive model comparison charts
   - Performance ranking and statistical significance

### Enhanced Libraries Used:
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **xgboost**: Gradient boosting framework (optional)
- **lightgbm**: Fast gradient boosting (optional)
- **plotly**: Interactive data visualization
- **seaborn**: Statistical data visualization
- **numpy**: Numerical computations

## 📊 Output

The enhanced script now generates:

### 🔬 **Model Comparison Analysis**:
1. **Algorithm Testing**: Evaluation of 4+ machine learning models
2. **Performance Comparison**: Side-by-side accuracy, F1-score, ROC-AUC comparison
3. **Cross-Validation Results**: Statistical validation with confidence intervals
4. **Best Model Selection**: Automatic winner identification and deployment
5. **Performance Ranking**: Comprehensive ranking with statistical significance

### 📊 **Standard Output**:
6. **Dataset Loading Summary**: Clean data confirmation
7. **Encoding Information**: Categorical variable mappings
8. **Performance Metrics**: Detailed classification report for best model
9. **ROC-AUC Score**: Overall discrimination capability
10. **Feature Importance Plot**: Interactive visualization for winner model
11. **Top Features Ranking**: Most predictive behaviors
12. **Emotion Label Mapping**: Class encoding reference

### 🎭 **Prediction Demonstrations**:
13. **Sample Predictions**: Demo predictions for various user types
14. **Interactive System**: User input prompts for real-time predictions

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
- **Dataset**: [Social Media Usage and Emotional Well-being](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being)
- **Author**: emirhanai
- **Original URL**: https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being
- **License**: Public domain / Open source (as per Kaggle terms)
- **Access Date**: November 2025

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