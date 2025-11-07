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

print(f"\n🎉 Social Platform Emotion Prediction Complete!")
print(f"Your AI model successfully predicts emotions from social media behavior!")
print(f"="*50)