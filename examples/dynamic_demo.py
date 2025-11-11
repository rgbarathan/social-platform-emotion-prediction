"""
Dynamic User Generation Demo
============================

This script demonstrates the dynamic random user generation feature
that creates new, realistic social media user profiles each time it runs.

Run this multiple times to see different users generated!
"""

import random
import datetime
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Loading AI Model (this may take a moment)...")

# Load the model and functions from main script
exec(open('Social Platform.py').read())

print("\n" + "="*70)
print("🎲 DYNAMIC USER GENERATION SHOWCASE")
print("="*70)
print(f"🕒 Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🔄 Run this script multiple times to see different users!")

def showcase_dynamic_generation():
    """Show multiple generations to demonstrate variety"""
    
    print(f"\n🎯 GENERATION SHOWCASE - 3 Different Runs:")
    print("="*60)
    
    for run in range(3):
        print(f"\n📍 Run #{run+1}:")
        print("-" * 20)
        
        # Generate 2 users for each run
        for i in range(2):
            user = generate_random_user()
            
            emotion, confidence = predict_user_emotion(
                user['age'], user['gender'], user['prediction_platform'],
                user['daily_usage'], user['posts'], user['likes'], 
                user['comments'], user['messages']
            )
            
            confidence_emoji = "🎯" if confidence >= 80 else "📈" if confidence >= 60 else "⚖️" if confidence >= 40 else "❓"
            
            print(f"  👤 {user['name']}")
            print(f"     🔸 {user['age']}yr {user['gender']} | {user['daily_usage']}min/day | {user['posts']} posts")
            print(f"     {confidence_emoji} {emotion} ({confidence:.1f}% confidence)")
        
        if run < 2:  # Small delay between runs for timestamp difference
            import time
            time.sleep(0.1)

def compare_platform_patterns():
    """Show how different platforms generate different user patterns"""
    
    print(f"\n🌐 PLATFORM-SPECIFIC USER PATTERNS:")
    print("="*60)
    
    target_platforms = ['Instagram', 'LinkedIn', 'TikTok', 'Facebook', 'Snapchat']
    
    for platform in target_platforms:
        print(f"\n📱 {platform} Users:")
        print("-" * 15)
        
        # Generate 3 users and keep only those from target platform
        attempts = 0
        platform_users = 0
        
        while platform_users < 2 and attempts < 20:
            user = generate_random_user()
            attempts += 1
            
            if user['display_platform'] == platform:
                platform_users += 1
                
                emotion, confidence = predict_user_emotion(
                    user['age'], user['gender'], user['prediction_platform'],
                    user['daily_usage'], user['posts'], user['likes'], 
                    user['comments'], user['messages']
                )
                
                print(f"  • {user['user_type']}: {user['age']}yr {user['gender']}")
                print(f"    {user['daily_usage']}min/day, {user['likes']} likes → {emotion} ({confidence:.1f}%)")

def extreme_user_showcase():
    """Generate and show some extreme user patterns"""
    
    print(f"\n🔥 EXTREME USER PATTERNS:")
    print("="*50)
    
    extreme_scenarios = [
        {
            'name': 'Social Media Addict',
            'modifier': lambda u: {**u, 'daily_usage': 400, 'posts': 15, 'likes': 500}
        },
        {
            'name': 'Digital Minimalist', 
            'modifier': lambda u: {**u, 'daily_usage': 10, 'posts': 0, 'likes': 2}
        },
        {
            'name': 'Viral Content Creator',
            'modifier': lambda u: {**u, 'posts': 20, 'likes': 1000, 'comments': 150}
        }
    ]
    
    for scenario in extreme_scenarios:
        base_user = generate_random_user()
        extreme_user = scenario['modifier'](base_user)
        
        emotion, confidence = predict_user_emotion(
            extreme_user['age'], extreme_user['gender'], extreme_user['prediction_platform'],
            extreme_user['daily_usage'], extreme_user['posts'], extreme_user['likes'], 
            extreme_user['comments'], extreme_user['messages']
        )
        
        print(f"\n💥 {scenario['name']}:")
        print(f"   Platform: {extreme_user['display_platform']}")
        print(f"   Pattern: {extreme_user['daily_usage']}min, {extreme_user['posts']}posts, {extreme_user['likes']}likes")
        print(f"   Result: {emotion} ({confidence:.1f}% confidence)")

# Run the showcase
showcase_dynamic_generation()
compare_platform_patterns()
extreme_user_showcase()

print(f"\n✨ DYNAMIC FEATURES SUMMARY:")
print("="*50)
print("🎲 Every run creates completely new users")
print("🎯 Realistic platform-specific behavior patterns") 
print("📊 Diverse age, gender, and usage combinations")
print("🔄 8 different platforms with unique characteristics")
print("⚡ Generated in real-time with timestamp")
print("🎭 Endless variety for testing and demonstration")

print(f"\n🎉 Try running this script multiple times to see the variety!")
print("="*70)