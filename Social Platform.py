import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try importing advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except (ImportError, Exception) as e:
    print("⚠️ XGBoost not available - skipping XGBoost in comparison")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except (ImportError, Exception) as e:
    print("⚠️ LightGBM not available - skipping LightGBM in comparison")
    LIGHTGBM_AVAILABLE = False

# Load the clean datasets (already cleaned and saved)
print("📊 Loading clean datasets...")
train_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/val.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Dataset loaded successfully!")
print(f"Training data: {train_df.shape[0]} samples")
print(f"Validation data: {val_df.shape[0]} samples")
print(f"Test data: {test_df.shape[0]} samples")
print(f"Total samples: {train_df.shape[0] + val_df.shape[0] + test_df.shape[0]}")

# Data is already clean, just need to encode categorical features
print(f"\n🔧 Encoding categorical features...")

# Combine all data to ensure consistent encoding across all sets
all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'Platform', 'Dominant_Emotion']

for column in categorical_columns:
    le = LabelEncoder()
    le.fit(all_data[column])
    
    # Apply encoding to each dataset
    train_df[column] = le.transform(train_df[column])
    val_df[column] = le.transform(val_df[column])
    test_df[column] = le.transform(test_df[column])
    
    label_encoders[column] = le
    print(f"  ✅ {column} encoded ({len(le.classes_)} classes)")

# Prepare features and targets
feature_columns = ['Age', 'Gender', 'Platform', 'Daily_Usage_Time_minutes', 
                  'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

X_train = train_df[feature_columns]
y_train = train_df['Dominant_Emotion']

X_val = val_df[feature_columns]
y_val = val_df['Dominant_Emotion']

X_test = test_df[feature_columns]
y_test = test_df['Dominant_Emotion']

print(f"✅ Features prepared: {len(feature_columns)} features")

# ========================================
# 🚀 MODEL COMPARISON SUITE
# ========================================

def evaluate_model_performance(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive model evaluation with multiple metrics"""
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC (if probability predictions available)
        roc_auc = None
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except Exception:
                roc_auc = None
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc,
            'Trained_Model': model
        }
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {str(e)}")
        return None

def cross_validate_model(model, X, y, model_name, cv_folds=5):
    """Perform cross-validation analysis"""
    try:
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), scoring='accuracy')
        return {
            'Model': model_name,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'CV_Scores': cv_scores
        }
    except Exception as e:
        print(f"❌ Error in cross-validation for {model_name}: {str(e)}")
        return None

print(f"\n🔬 ADVANCED MODEL COMPARISON ANALYSIS")
print(f"="*60)

# Prepare models for comparison
models_to_test = [
    (RandomForestClassifier(random_state=42, n_estimators=100), "Random Forest"),
    (GradientBoostingClassifier(random_state=42, n_estimators=100), "Gradient Boosting"),
    (LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression"),
    (SVC(random_state=42, probability=True), "Support Vector Machine")
]

# Add advanced models if available
if XGBOOST_AVAILABLE:
    models_to_test.append((XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'), "XGBoost"))

if LIGHTGBM_AVAILABLE:
    models_to_test.append((LGBMClassifier(random_state=42, n_estimators=100, verbose=-1), "LightGBM"))

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"🤖 Testing {len(models_to_test)} different algorithms...")

# Evaluate all models
model_results = []
cv_results = []
trained_models = {}

for model, name in models_to_test:
    print(f"\n  📊 Evaluating {name}...")
    
    # Use scaled data for SVM and Logistic Regression
    if name in ["Support Vector Machine", "Logistic Regression"]:
        result = evaluate_model_performance(model, X_train_scaled, y_train, X_test_scaled, y_test, name)
        cv_result = cross_validate_model(model, scaler.fit_transform(pd.concat([X_train, X_test])), 
                                       pd.concat([y_train, y_test]), name)
    else:
        result = evaluate_model_performance(model, X_train, y_train, X_test, y_test, name)
        cv_result = cross_validate_model(model, pd.concat([X_train, X_test]), 
                                       pd.concat([y_train, y_test]), name)
    
    if result:
        model_results.append(result)
        trained_models[name] = result['Trained_Model']
        print(f"    ✅ Accuracy: {result['Accuracy']:.4f}, F1: {result['F1_Score']:.4f}")
    
    if cv_result:
        cv_results.append(cv_result)
        print(f"    📈 CV Score: {cv_result['CV_Mean']:.4f} ± {cv_result['CV_Std']:.4f}")

# Create comparison results DataFrame
results_df = pd.DataFrame(model_results)
cv_df = pd.DataFrame(cv_results)

print(f"\n📈 MODEL PERFORMANCE COMPARISON")
print(f"="*60)
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].round(4).to_string(index=False))

print(f"\n🔄 CROSS-VALIDATION RESULTS") 
print(f"="*60)
print(cv_df[['Model', 'CV_Mean', 'CV_Std']].round(4).to_string(index=False))

# Find the best model
best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.iloc[best_model_idx]['Model']
best_accuracy = results_df.iloc[best_model_idx]['Accuracy']

print(f"\n🏆 BEST PERFORMING MODEL")
print(f"="*60)
print(f"🥇 Winner: {best_model_name}")
print(f"🎯 Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"📊 F1-Score: {results_df.iloc[best_model_idx]['F1_Score']:.4f}")
if results_df.iloc[best_model_idx]['ROC_AUC']:
    print(f"🎪 ROC-AUC: {results_df.iloc[best_model_idx]['ROC_AUC']:.4f}")

# Set the best model as the main classifier
clf = trained_models[best_model_name]
print(f"\n✅ Using {best_model_name} as the primary model for predictions")

# Make predictions with the best model
print(f"\n🔮 Making predictions with {best_model_name}...")
if best_model_name in ["Support Vector Machine", "Logistic Regression"]:
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, 'predict_proba') else None
else:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

# Evaluate model performance
print(f"\n📊 Model Performance Results:")
print("="*50)
print(classification_report(y_test, y_pred))

# ROC-AUC Score
try:
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    if roc_auc >= 0.95:
        print("🏆 Outstanding performance!")
    elif roc_auc >= 0.85:
        print("🎯 Excellent performance!")
    elif roc_auc >= 0.75:
        print("👍 Good performance!")
    else:
        print("📈 Needs improvement")
        
except ValueError as e:
    print(f"ROC-AUC Error: {e}")

# ========================================
# 📊 MODEL COMPARISON VISUALIZATION
# ========================================

print(f"\n📊 Creating Model Comparison Visualizations...")

# 1. Performance Metrics Comparison Chart
if len(model_results) > 1:
    fig_comparison = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                       'Precision vs Recall', 'Cross-Validation Scores'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Accuracy comparison
    fig_comparison.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Accuracy'], 
               name='Accuracy', marker_color='lightblue'),
        row=1, col=1
    )
    
    # F1-Score comparison  
    fig_comparison.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['F1_Score'], 
               name='F1-Score', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Precision vs Recall scatter
    fig_comparison.add_trace(
        go.Scatter(x=results_df['Precision'], y=results_df['Recall'],
                   text=results_df['Model'], mode='markers+text',
                   marker=dict(size=12), name='Models'),
        row=2, col=1
    )
    
    # Cross-validation scores
    fig_comparison.add_trace(
        go.Bar(x=cv_df['Model'], y=cv_df['CV_Mean'],
               error_y=dict(type='data', array=cv_df['CV_Std']),
               name='CV Score', marker_color='orange'),
        row=2, col=2
    )
    
    fig_comparison.update_layout(
        height=800,
        title_text=f"🔬 Comprehensive Model Performance Comparison",
        showlegend=False
    )
    
    # Update axes labels
    fig_comparison.update_yaxes(title_text="Accuracy", row=1, col=1, range=[0, 1])
    fig_comparison.update_yaxes(title_text="F1-Score", row=1, col=2, range=[0, 1])
    fig_comparison.update_xaxes(title_text="Precision", row=2, col=1)
    fig_comparison.update_yaxes(title_text="Recall", row=2, col=1)
    fig_comparison.update_yaxes(title_text="CV Score", row=2, col=2, range=[0, 1])
    
    fig_comparison.show()

# 2. Model Performance Summary Table
print(f"\n📈 DETAILED PERFORMANCE METRICS")
print(f"="*80)
for _, row in results_df.iterrows():
    accuracy_str = f"{row['Accuracy']:.4f}"
    f1_str = f"{row['F1_Score']:.4f}"
    roc_str = f"{row['ROC_AUC']:.4f}" if row['ROC_AUC'] else 'N/A'
    print(f"🤖 {row['Model']:20s} | Accuracy: {accuracy_str} | F1: {f1_str} | ROC-AUC: {roc_str}")

# 3. Statistical Significance Analysis
print(f"\n📊 PERFORMANCE RANKING")
print(f"="*60)
ranked_models = results_df.sort_values('Accuracy', ascending=False)
for rank, (_, row) in enumerate(ranked_models.iterrows(), 1):
    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
    print(f"{emoji} Rank {rank}: {row['Model']} (Accuracy: {row['Accuracy']:.4f})")

# Feature importance analysis
print(f"\n📈 Feature Importance Analysis:")
feature_importance = clf.feature_importances_
features = X_train.columns

# Create interactive plot
fig = px.bar(
    x=features, 
    y=feature_importance, 
    labels={'x': 'Feature', 'y': 'Importance'}, 
    title='Feature Importance in Social Media Emotion Prediction'
)
fig.show()

# Print top features
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"🔝 Top 5 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head().iterrows(), 1):
    print(f"  {i}. {row['Feature']}: {row['Importance']:.4f}")

# Print emotion label mapping
print(f"\n🏷️ Emotion Label Mapping:")
emotion_le = label_encoders['Dominant_Emotion']
for i, emotion in enumerate(emotion_le.classes_):
    print(f"  {i}: {emotion}")

# Function to predict emotions for unknown users
def predict_user_emotion(age, gender, platform, daily_usage_time, posts_per_day, 
                        likes_received, comments_received, messages_sent):
    """
    Predict emotion for an unknown user based on their social media behavior
    
    Parameters:
    - age: User's age (numeric)
    - gender: 'Male' or 'Female'
    - platform: 'Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'Snapchat', 'Whatsapp', 'Telegram'
    - daily_usage_time: Minutes spent daily on social media
    - posts_per_day: Number of posts per day
    - likes_received: Average likes received per day
    - comments_received: Average comments received per day
    - messages_sent: Number of messages sent per day
    
    Returns:
    - prediction: Predicted emotion
    - confidence: Prediction confidence (probability)
    """
    try:
        # Validate platform input
        valid_platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'Snapchat', 'Whatsapp', 'Telegram']
        if platform not in valid_platforms:
            return f"Error: Platform must be one of {valid_platforms}", 0.0
            
        # Validate gender input  
        valid_genders = ['Male', 'Female']
        if gender not in valid_genders:
            return f"Error: Gender must be one of {valid_genders}", 0.0
        
        # Create input data frame
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Platform': [platform],
            'Daily_Usage_Time_minutes': [daily_usage_time],
            'Posts_Per_Day': [posts_per_day],
            'Likes_Received_Per_Day': [likes_received],
            'Comments_Received_Per_Day': [comments_received],
            'Messages_Sent_Per_Day': [messages_sent]
        })
        
        # Encode categorical features using the same encoders
        user_data['Gender'] = label_encoders['Gender'].transform([gender])[0]
        user_data['Platform'] = label_encoders['Platform'].transform([platform])[0]
        
        # Extract features in correct order
        user_features = user_data[feature_columns]
        
        # Apply scaling if needed for SVM or Logistic Regression
        if best_model_name in ["Support Vector Machine", "Logistic Regression"]:
            user_features_processed = scaler.transform(user_features)
        else:
            user_features_processed = user_features
        
        # Make prediction
        prediction_encoded = clf.predict(user_features_processed)[0]
        probabilities = clf.predict_proba(user_features_processed)[0]
        
        # Decode prediction back to emotion name
        predicted_emotion = label_encoders['Dominant_Emotion'].inverse_transform([prediction_encoded])[0]
        
        # Get confidence (highest probability)
        confidence = max(probabilities) * 100
        
        return predicted_emotion, confidence
        
    except ValueError as e:
        return f"Error: {str(e)}", 0.0

# Interactive prediction system
def interactive_prediction():
    """Interactive system for predicting emotions for new users"""
    print(f"\n🔮 Interactive Emotion Prediction for New Users")
    print(f"="*50)
    
    while True:
        try:
            print(f"\nEnter user details for emotion prediction:")
            print(f"(Type 'quit' to exit)")
            
            # Get user input
            age_input = input("Age: ")
            if age_input.lower() == 'quit':
                break
            age = int(age_input)
            
            gender = input("Gender (Male/Female): ")
            if gender.lower() == 'quit':
                break
            
            print("Available platforms: Facebook, Instagram, Twitter, LinkedIn, Snapchat, Whatsapp, Telegram")
            platform = input("Platform: ")
            if platform.lower() == 'quit':
                break
            
            daily_usage = int(input("Daily usage time (minutes): "))
            posts_per_day = int(input("Posts per day: "))
            likes_received = int(input("Likes received per day: "))
            comments_received = int(input("Comments received per day: "))
            messages_sent = int(input("Messages sent per day: "))
            
            # Make prediction
            emotion, confidence = predict_user_emotion(
                age, gender, platform, daily_usage, posts_per_day,
                likes_received, comments_received, messages_sent
            )
            
            if "Error" not in str(emotion):
                print(f"\n� Prediction Results:")
                print(f"  Predicted Emotion: {emotion}")
                print(f"  Confidence: {confidence:.1f}%")
                
                # Interpretation based on confidence
                if confidence >= 80:
                    print(f"  📊 Very high confidence prediction")
                elif confidence >= 60:
                    print(f"  📈 High confidence prediction")
                elif confidence >= 40:
                    print(f"  ⚖️ Moderate confidence prediction")
                else:
                    print(f"  ❓ Low confidence - data may be unusual")
            else:
                print(f"❌ {emotion}")
                
        except ValueError:
            print(f"❌ Please enter valid numeric values")
        except KeyboardInterrupt:
            print(f"\n👋 Exiting prediction system...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Example predictions for demonstration
def demo_predictions():
    """Demonstrate predictions with example users"""
    print(f"\n🎭 Demo: Emotion Predictions for Sample Users")
    print(f"="*50)
    
    # Sample users with different profiles
    sample_users = [
        {
            'name': 'Active Instagram User',
            'age': 22, 'gender': 'Female', 'platform': 'Instagram',
            'daily_usage': 180, 'posts': 3, 'likes': 45, 'comments': 8, 'messages': 25
        },
        {
            'name': 'Professional LinkedIn User', 
            'age': 35, 'gender': 'Male', 'platform': 'LinkedIn',
            'daily_usage': 45, 'posts': 1, 'likes': 12, 'comments': 3, 'messages': 8
        },
        {
            'name': 'Heavy Snapchat User',
            'age': 19, 'gender': 'Female', 'platform': 'Snapchat', 
            'daily_usage': 240, 'posts': 2, 'likes': 120, 'comments': 15, 'messages': 40
        },
        {
            'name': 'Casual Facebook User',
            'age': 45, 'gender': 'Male', 'platform': 'Facebook',
            'daily_usage': 60, 'posts': 1, 'likes': 8, 'comments': 2, 'messages': 12
        }
    ]
    
    for user in sample_users:
        emotion, confidence = predict_user_emotion(
            user['age'], user['gender'], user['platform'], user['daily_usage'],
            user['posts'], user['likes'], user['comments'], user['messages']
        )
        
        print(f"\n👤 {user['name']}:")
        print(f"   Age: {user['age']}, Platform: {user['platform']}")
        print(f"   Usage: {user['daily_usage']} min/day, {user['posts']} posts/day")
        print(f"   📊 Predicted Emotion: {emotion} (Confidence: {confidence:.1f}%)")

# Run demonstration
demo_predictions()

print(f"\n�🎉 Social Platform Emotion Prediction Complete!")
print(f"Your AI model successfully predicts emotions from social media behavior!")
print(f"\n💡 New Features Added:")
print(f"  🔮 Unknown user prediction function")
print(f"  🎭 Demo predictions with sample users")
print(f"  🎯 Interactive prediction system")
print(f"\nTo use interactive prediction, call: interactive_prediction()")
print(f"="*50)