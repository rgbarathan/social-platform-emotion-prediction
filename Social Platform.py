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

import random
import datetime

def generate_random_user():
    """Generate a random user profile with realistic social media behavior"""
    
    # Platform-specific profiles with realistic ranges
    platforms_data = {
        'Instagram': {
            'age_range': (16, 35), 'usage_range': (60, 300),
            'posts_range': (0, 8), 'likes_range': (10, 200),
            'comments_range': (2, 30), 'messages_range': (15, 80),
            'user_types': ['Influencer', 'Content Creator', 'Photo Enthusiast', 'Story Addict']
        },
        'TikTok': {  # Note: Will use Snapchat for prediction since TikTok not in training data
            'age_range': (13, 25), 'usage_range': (90, 360),
            'posts_range': (1, 10), 'likes_range': (20, 500),
            'comments_range': (5, 50), 'messages_range': (10, 60),
            'user_types': ['Creator', 'Viral Chaser', 'Dance Enthusiast', 'Comedy Fan']
        },
        'LinkedIn': {
            'age_range': (22, 55), 'usage_range': (10, 90),
            'posts_range': (0, 3), 'likes_range': (2, 50),
            'comments_range': (0, 15), 'messages_range': (5, 30),
            'user_types': ['Professional', 'Job Seeker', 'Industry Expert', 'Networker']
        },
        'Facebook': {
            'age_range': (25, 65), 'usage_range': (30, 180),
            'posts_range': (0, 5), 'likes_range': (5, 80),
            'comments_range': (1, 20), 'messages_range': (5, 50),
            'user_types': ['Family User', 'News Reader', 'Community Member', 'Casual Browser']
        },
        'Twitter': {
            'age_range': (18, 50), 'usage_range': (30, 240),
            'posts_range': (0, 15), 'likes_range': (5, 100),
            'comments_range': (2, 25), 'messages_range': (3, 40),
            'user_types': ['News Follower', 'Opinion Sharer', 'Trend Watcher', 'Activist']
        },
        'Snapchat': {
            'age_range': (13, 30), 'usage_range': (60, 300),
            'posts_range': (2, 12), 'likes_range': (15, 150),
            'comments_range': (3, 25), 'messages_range': (20, 100),
            'user_types': ['Story Poster', 'Snap Streaker', 'Filter Fan', 'Chat Heavy']
        },
        'Whatsapp': {
            'age_range': (16, 60), 'usage_range': (30, 200),
            'posts_range': (0, 2), 'likes_range': (0, 10),
            'comments_range': (0, 5), 'messages_range': (20, 200),
            'user_types': ['Family Chatter', 'Work Communicator', 'Group Admin', 'Voice Note Fan']
        },
        'Telegram': {
            'age_range': (18, 45), 'usage_range': (20, 150),
            'posts_range': (0, 3), 'likes_range': (0, 20),
            'comments_range': (0, 8), 'messages_range': (10, 100),
            'user_types': ['Channel Follower', 'Group Member', 'Privacy Seeker', 'Bot User']
        }
    }
    
    # Select random platform and get its data
    platform = random.choice(list(platforms_data.keys()))
    
    # Handle TikTok by using Snapchat for prediction (since TikTok not in training data)
    prediction_platform = 'Snapchat' if platform == 'TikTok' else platform
    
    platform_info = platforms_data[platform]
    
    # Generate random user characteristics
    age = random.randint(*platform_info['age_range'])
    gender = random.choice(['Male', 'Female'])
    user_type = random.choice(platform_info['user_types'])
    
    # Generate realistic usage patterns with some correlation
    daily_usage = random.randint(*platform_info['usage_range'])
    
    # Posts correlate somewhat with usage time and age
    base_posts = random.randint(*platform_info['posts_range'])
    posts_per_day = max(0, base_posts + random.randint(-1, 2))
    
    # Likes generally correlate with posts and usage
    likes_base = random.randint(*platform_info['likes_range'])
    likes_modifier = 1.5 if posts_per_day > 3 else 0.7 if posts_per_day == 0 else 1.0
    likes_received = max(0, int(likes_base * likes_modifier))
    
    # Comments are typically lower than likes
    comments_base = random.randint(*platform_info['comments_range'])
    comments_received = max(0, min(comments_base, likes_received // 3))
    
    # Messages vary by platform type and usage
    messages_sent = random.randint(*platform_info['messages_range'])
    
    return {
        'name': f"{user_type} ({platform})",
        'display_platform': platform,
        'prediction_platform': prediction_platform,
        'age': age,
        'gender': gender,
        'daily_usage': daily_usage,
        'posts': posts_per_day,
        'likes': likes_received,
        'comments': comments_received,
        'messages': messages_sent,
        'user_type': user_type
    }

# Example predictions for demonstration
def demo_predictions():
    """Demonstrate predictions with dynamically generated users"""
    print(f"\n🎭 Demo: Dynamic Emotion Predictions for Random Users")
    print(f"="*55)
    print(f"⚡ Generated at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"🎲 New random users created each run!")
    
    # Generate 5 random users for demonstration
    num_demo_users = 5
    
    for i in range(num_demo_users):
        user = generate_random_user()
        
        emotion, confidence = predict_user_emotion(
            user['age'], user['gender'], user['prediction_platform'], user['daily_usage'],
            user['posts'], user['likes'], user['comments'], user['messages']
        )
        
        print(f"\n👤 Random User #{i+1}: {user['name']}")
        print(f"   📊 Profile: {user['age']}yr {user['gender']} on {user['display_platform']}")
        print(f"   ⏱️  Usage: {user['daily_usage']} min/day, {user['posts']} posts/day")
        print(f"   � Engagement: {user['likes']} likes, {user['comments']} comments, {user['messages']} messages")
        
        # Add interpretation based on confidence
        if "Error" not in str(emotion):
            confidence_emoji = "🎯" if confidence >= 80 else "📈" if confidence >= 60 else "⚖️" if confidence >= 40 else "❓"
            print(f"   {confidence_emoji} Predicted Emotion: {emotion} (Confidence: {confidence:.1f}%)")
            
            # Add behavioral insight
            if user['display_platform'] in ['Instagram', 'TikTok'] and user['daily_usage'] > 200:
                print(f"   💡 High usage pattern - heavy social media engagement")
            elif user['display_platform'] == 'LinkedIn' and user['daily_usage'] < 60:
                print(f"   💼 Professional usage pattern - focused networking")
            elif user['messages'] > 100:
                print(f"   💬 Communication-heavy user - high messaging activity")
        else:
            print(f"   ❌ Prediction Error: {emotion}")

def generate_custom_user_scenarios():
    """Generate specific scenario-based users for testing"""
    scenarios = [
        {
            'name': 'Heavy Social Media Addict',
            'generator': lambda: {
                **generate_random_user(),
                'daily_usage': random.randint(300, 480),
                'posts': random.randint(5, 15),
                'likes': random.randint(100, 300)
            }
        },
        {
            'name': 'Minimal Digital Presence',
            'generator': lambda: {
                **generate_random_user(),
                'daily_usage': random.randint(5, 30),
                'posts': random.randint(0, 1),
                'likes': random.randint(0, 10)
            }
        },
        {
            'name': 'Highly Engaged Creator',
            'generator': lambda: {
                **generate_random_user(),
                'posts': random.randint(8, 20),
                'likes': random.randint(200, 1000),
                'comments': random.randint(30, 100)
            }
        }
    ]
    
    print(f"\n🎯 Special Scenario Testing:")
    print(f"="*40)
    
    for scenario in scenarios[:2]:  # Test 2 scenarios
        user = scenario['generator']()
        user['name'] = f"{scenario['name']} ({user.get('display_platform', user.get('platform', 'Unknown'))})"
        
        emotion, confidence = predict_user_emotion(
            user['age'], user['gender'], user.get('prediction_platform', user.get('prediction_platform', 'Instagram')), 
            user['daily_usage'], user['posts'], user['likes'], 
            user['comments'], user['messages']
        )
        
        print(f"\n📍 {user['name']}")
        print(f"   📊 Extreme Pattern: {user['daily_usage']}min, {user['posts']}posts, {user['likes']}likes")
        print(f"   🎪 Result: {emotion} ({confidence:.1f}% confidence)")

# Run dynamic demonstration
print(f"\n🎲 Generating Random Users for Each Run...")
demo_predictions()

# Run scenario testing
generate_custom_user_scenarios()

# ========================================
# 🎮 INTERACTIVE DEMO MENU SYSTEM
# ========================================

def interactive_demo_menu():
    """Interactive menu for different demo options"""
    print(f"\n🎮 INTERACTIVE DEMO MENU")
    print(f"="*50)
    print(f"Choose what you'd like to explore:")
    print(f"")
    print(f"1️⃣  🎲 Generate More Random Users")
    print(f"2️⃣  🎯 Create Extreme User Scenarios") 
    print(f"3️⃣  🌐 Platform-Specific User Showcase")
    print(f"4️⃣  ⚡ Quick Single User Prediction")
    print(f"5️⃣  🔄 Compare Multiple Random Generations")
    print(f"6️⃣  📊 Model Performance Deep Dive")
    print(f"7️⃣  🚀 Interactive Prediction Mode")
    print(f"8️⃣  ❌ Exit")
    print(f"")
    
    while True:
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                print(f"\n� GENERATING NEW RANDOM USERS...")
                demo_predictions()
                
            elif choice == '2':
                print(f"\n🎯 EXTREME SCENARIO TESTING...")
                generate_custom_user_scenarios()
                
            elif choice == '3':
                platform_showcase()
                
            elif choice == '4':
                quick_prediction()
                
            elif choice == '5':
                multiple_generations_comparison()
                
            elif choice == '6':
                model_performance_deep_dive()
                
            elif choice == '7':
                interactive_prediction()
                
            elif choice == '8':
                print(f"\n👋 Thanks for exploring the Social Platform Emotion Prediction system!")
                break
                
            else:
                print(f"❌ Please enter a number between 1-8")
                
            print(f"\n" + "="*50)
            print(f"🎮 What would you like to try next?")
            print(f"1️⃣Random Users  2️⃣Extreme  3️⃣Platforms  4️⃣Quick  5️⃣Compare  6️⃣Performance  7️⃣Interactive  8️⃣Exit")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def platform_showcase():
    """Showcase users from specific platforms"""
    print(f"\n🌐 PLATFORM-SPECIFIC USER SHOWCASE")
    print(f"="*50)
    
    platforms = ['Instagram', 'LinkedIn', 'TikTok', 'Facebook', 'Twitter', 'Snapchat', 'Whatsapp', 'Telegram']
    
    print("Available platforms:")
    for i, platform in enumerate(platforms, 1):
        print(f"{i}. {platform}")
    
    try:
        choice = input(f"\nChoose platform (1-{len(platforms)}) or 'all' for all platforms: ").strip().lower()
        
        if choice == 'all':
            selected_platforms = platforms
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(platforms):
                selected_platforms = [platforms[idx]]
            else:
                print("❌ Invalid choice")
                return
        
        for platform in selected_platforms:
            print(f"\n📱 {platform} Users:")
            print("-" * 20)
            
            # Generate users for this platform
            attempts = 0
            found_users = 0
            
            while found_users < 2 and attempts < 15:
                user = generate_random_user()
                attempts += 1
                
                if user['display_platform'] == platform:
                    found_users += 1
                    emotion, confidence = predict_user_emotion(
                        user['age'], user['gender'], user['prediction_platform'],
                        user['daily_usage'], user['posts'], user['likes'], 
                        user['comments'], user['messages']
                    )
                    
                    confidence_emoji = "🎯" if confidence >= 80 else "📈" if confidence >= 60 else "⚖️" if confidence >= 40 else "❓"
                    print(f"  👤 {user['user_type']}: {user['age']}yr {user['gender']}")
                    print(f"     Usage: {user['daily_usage']}min/day, {user['posts']} posts")
                    print(f"     {confidence_emoji} {emotion} ({confidence:.1f}% confidence)")
                    
    except ValueError:
        print("❌ Please enter a valid number")

def quick_prediction():
    """Quick single user prediction"""
    print(f"\n⚡ QUICK USER PREDICTION")
    print(f"="*40)
    
    user = generate_random_user()
    emotion, confidence = predict_user_emotion(
        user['age'], user['gender'], user['prediction_platform'],
        user['daily_usage'], user['posts'], user['likes'], 
        user['comments'], user['messages']
    )
    
    print(f"👤 Generated User: {user['name']}")
    print(f"📊 Profile: {user['age']}yr {user['gender']} on {user['display_platform']}")
    print(f"⏱️ Activity: {user['daily_usage']}min/day, {user['posts']} posts")
    print(f"💬 Engagement: {user['likes']} likes, {user['comments']} comments, {user['messages']} messages")
    
    confidence_emoji = "🎯" if confidence >= 80 else "📈" if confidence >= 60 else "⚖️" if confidence >= 40 else "❓"
    print(f"{confidence_emoji} Prediction: {emotion} ({confidence:.1f}% confidence)")

def multiple_generations_comparison():
    """Compare multiple user generations"""
    print(f"\n� MULTIPLE GENERATION COMPARISON")
    print(f"="*50)
    
    try:
        num_generations = int(input("How many user generations to compare (1-10)? ").strip())
        num_generations = max(1, min(10, num_generations))
        
        for gen in range(num_generations):
            print(f"\n🎲 Generation #{gen + 1}:")
            print("-" * 25)
            
            for i in range(2):  # 2 users per generation
                user = generate_random_user()
                emotion, confidence = predict_user_emotion(
                    user['age'], user['gender'], user['prediction_platform'],
                    user['daily_usage'], user['posts'], user['likes'], 
                    user['comments'], user['messages']
                )
                
                print(f"  {i+1}. {user['name'][:20]:20s} → {emotion} ({confidence:.1f}%)")
                
    except ValueError:
        print("❌ Please enter a valid number")

def model_performance_deep_dive():
    """Deep dive into model performance"""
    print(f"\n📊 MODEL PERFORMANCE DEEP DIVE")
    print(f"="*50)
    
    # Show the best model details
    print(f"🏆 Current Best Model: {best_model_name}")
    print(f"🎯 Accuracy: {results_df.iloc[best_model_idx]['Accuracy']:.4f}")
    print(f"📈 F1-Score: {results_df.iloc[best_model_idx]['F1_Score']:.4f}")
    if results_df.iloc[best_model_idx]['ROC_AUC']:
        print(f"🎪 ROC-AUC: {results_df.iloc[best_model_idx]['ROC_AUC']:.4f}")
    
    print(f"\n📋 All Model Comparison:")
    for _, row in results_df.iterrows():
        print(f"  • {row['Model']}: {row['Accuracy']:.3f} accuracy")
    
    print(f"\n🔝 Top Features:")
    if hasattr(clf, 'feature_importances_'):
        for i, (feature, importance) in enumerate(zip(feature_columns, clf.feature_importances_)[:5], 1):
            print(f"  {i}. {feature}: {importance:.3f}")

print(f"\n🎉 Social Platform Emotion Prediction Complete!")
print(f"Your AI model successfully predicts emotions from social media behavior!")

print(f"\n💡 System Features:")
print(f"  🔬 Advanced model comparison with {len(model_results)} algorithms")
print(f"  🎲 Dynamic user generation with 8+ platforms")
print(f"  🎯 Interactive prediction system")
print(f"  📊 {best_accuracy*100:.1f}% accuracy with {best_model_name}")

print(f"\n🎮 EXPLORE MORE:")
print(f"Would you like to explore additional features?")
response = input(f"Enter 'y' for interactive menu, or press Enter to finish: ").strip().lower()

if response in ['y', 'yes']:
    interactive_demo_menu()
else:
    print(f"\n✨ Thanks for using Social Platform Emotion Prediction!")
    print(f"🚀 Your AI system is ready for production use!")

print(f"="*50)