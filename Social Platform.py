import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

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

# Train Random Forest model
print(f"\n🤖 Training Random Forest model...")
clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
print(f"✅ Model trained successfully!")

# Make predictions
print(f"\n🔮 Making predictions...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

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
        
        # Make prediction
        prediction_encoded = clf.predict(user_features)[0]
        probabilities = clf.predict_proba(user_features)[0]
        
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