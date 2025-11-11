"""
Example Usage of Unknown User Prediction System
===============================================

This file demonstrates how to use the emotion prediction system 
for new/unknown users not in the training dataset.
"""

# Import the main script to access the trained model and functions
import sys
import os

# Add the current directory to path to import from Social Platform.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First, we need to run the main script to train the model
print("🚀 Setting up the AI model...")
exec(open('Social Platform.py').read())

print("\n" + "="*60)
print("📱 UNKNOWN USER EMOTION PREDICTION EXAMPLES")
print("="*60)

# Dynamic Examples Section
print("\n🎲 DYNAMIC RANDOM USER EXAMPLES")
print("-" * 50)
print("⚡ These users are randomly generated each time you run this script!")

for i in range(3):
    print(f"\n🎯 Dynamic Example #{i+1}:")
    user = generate_random_user()
    
    emotion, confidence = predict_user_emotion(
        user['age'], user['gender'], user['prediction_platform'],
        user['daily_usage'], user['posts'], user['likes'], 
        user['comments'], user['messages']
    )
    
    print(f"Profile: {user['age']}yr {user['gender']} {user['name']}")
    print(f"Usage: {user['daily_usage']}min/day, {user['posts']} posts, {user['likes']} likes, {user['messages']} messages")
    print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

# Static Examples Section (for comparison)
print("\n📊 STATIC REFERENCE EXAMPLES")
print("-" * 50)

# Example 1: Predict for a new Instagram influencer
print("\n📸 Example 1: Instagram Influencer")
print("-" * 30)
emotion, confidence = predict_user_emotion(
    age=24,
    gender='Female', 
    platform='Instagram',
    daily_usage_time=240,  # 4 hours daily
    posts_per_day=3,
    likes_received=150,
    comments_received=25,
    messages_sent=40
)
print(f"Profile: 24yr Female Instagram user (240min/day, 3 posts, 150 likes)")
print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

# Example 2: Corporate LinkedIn user
print("\n💼 Example 2: Corporate Professional")
print("-" * 30)
emotion, confidence = predict_user_emotion(
    age=45,
    gender='Male',
    platform='LinkedIn', 
    daily_usage_time=20,   # 20 minutes daily
    posts_per_day=1,
    likes_received=8,
    comments_received=2,
    messages_sent=5
)
print(f"Profile: 45yr Male LinkedIn user (20min/day, 1 post, 8 likes)")
print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

# Example 3: Teen social media user
print("\n🧑‍🎓 Example 3: Teen Social Media User")
print("-" * 30)
emotion, confidence = predict_user_emotion(
    age=17,
    gender='Male',
    platform='Snapchat',
    daily_usage_time=180,  # 3 hours daily
    posts_per_day=5,
    likes_received=45,
    comments_received=12,
    messages_sent=80
)
print(f"Profile: 17yr Male Snapchat user (180min/day, 5 posts, 45 likes)")
print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

# Example 4: Minimal social media user
print("\n😐 Example 4: Minimal Social Media User")
print("-" * 30)
emotion, confidence = predict_user_emotion(
    age=35,
    gender='Female',
    platform='Facebook',
    daily_usage_time=30,   # 30 minutes daily
    posts_per_day=0,       # No posts
    likes_received=2,
    comments_received=0,
    messages_sent=8
)
print(f"Profile: 35yr Female Facebook user (30min/day, 0 posts, 2 likes)")
print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

# Example 5: Heavy messaging user
print("\n💬 Example 5: Heavy Messaging User")
print("-" * 30)
emotion, confidence = predict_user_emotion(
    age=28,
    gender='Female',
    platform='Whatsapp',
    daily_usage_time=90,
    posts_per_day=0,       # No posts (messaging only)
    likes_received=0,
    comments_received=0,
    messages_sent=150      # Heavy messaging
)
print(f"Profile: 28yr Female WhatsApp user (90min/day, 0 posts, 150 messages)")
print(f"Prediction: {emotion} (Confidence: {confidence:.1f}%)")

print("\n" + "="*60)
print("🎯 INTERPRETATION GUIDE")
print("="*60)
print("Confidence Levels:")
print("  🟢 80-100%: Very High Confidence")
print("  🟡 60-79%:  High Confidence") 
print("  🟠 40-59%:  Moderate Confidence")
print("  🔴 <40%:    Low Confidence (unusual pattern)")
print("\nEmotion Categories:")
print("  😡 Anger/Aggression: Negative, hostile emotions")
print("  😰 Anxiety: Worried, stressed feelings") 
print("  😴 Boredom: Disengaged, uninterested")
print("  😊 Happiness: Positive, joyful emotions")
print("  😐 Neutral: Balanced emotional state")
print("  😢 Sadness: Low mood, melancholy")

print("\n" + "="*60)
print("💡 USAGE TIPS")
print("="*60)
print("1. Higher usage time often correlates with stronger emotions")
print("2. More social interaction (likes/comments) tends toward happiness")
print("3. Professional platforms (LinkedIn) often show neutral/boredom")
print("4. Low confidence may indicate unusual usage patterns")
print("5. Consider multiple factors when interpreting results")

print("\n🎉 Try the interactive mode by calling: interactive_prediction()")