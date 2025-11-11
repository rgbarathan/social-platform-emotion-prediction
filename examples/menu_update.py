# Simplified Interactive Menu Function

def interactive_demo_menu():
    """Simplified interactive menu with essential options only"""
    print(f"\n🎮 INTERACTIVE DEMO MENU")
    print(f"="*50)
    print(f"Choose what you'd like to explore:")
    print(f"")
    print(f"1️⃣  🎲 Generate New Random Users")
    print(f"2️⃣  🎯 Test Your Own User Data") 
    print(f"3️⃣  📊 View Model Performance")
    print(f"4️⃣  🔬 Advanced Model Comparison")
    print(f"5️⃣  ❌ Exit")
    print(f"")
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                print(f"\n🎲 GENERATING NEW RANDOM USERS...")
                demo_predictions()
                
            elif choice == '2':
                print(f"\n🎯 INTERACTIVE USER INPUT...")
                interactive_prediction()
                
            elif choice == '3':
                model_performance_deep_dive()
                
            elif choice == '4':
                print(f"\n🔬 RUNNING COMPREHENSIVE MODEL COMPARISON...")
                run_full_model_comparison()
                
            elif choice == '5':
                print(f"\n👋 Thanks for exploring the Social Platform Emotion Prediction system!")
                break
                
            else:
                print(f"❌ Please enter a number between 1-5")
                
            print(f"\n" + "="*50)
            print(f"🎮 What would you like to try next?")
            print(f"1️⃣Generate  2️⃣Test  3️⃣Performance  4️⃣Compare  5️⃣Exit")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")