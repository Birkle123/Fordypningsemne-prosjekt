#!/usr/bin/env python3
"""
Helper script to clean up old Benders decomposition files
Keep only Project_model.py and this cleanup script
"""

import os
import shutil

# Files to remove (keep only main script)
files_to_remove = [
    "benders_decomposition.py",
    "debug_feasibility.py", 
    "debug_subproblem.py",
    "finite_diff_benders.py",
    "improved_benders.py",
    "marginal_value_analysis.py",
    "test_constraint_impact.py",
    "trace_scenario5.py"
]

def cleanup_files():
    """Remove all the extra scripts created during development"""
    print("🧹 Cleaning up extra scripts...")
    print("Keeping: Project_model.py, cleanup_helper.py")
    print()
    
    removed_count = 0
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"✅ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")
        else:
            print(f"ℹ️  Not found: {filename}")
    
    # Also remove __pycache__ if it exists
    if os.path.exists("__pycache__"):
        try:
            shutil.rmtree("__pycache__")
            print("✅ Removed: __pycache__/")
            removed_count += 1
        except Exception as e:
            print(f"❌ Failed to remove __pycache__: {e}")
    
    print()
    print(f"🎯 Cleanup complete! Removed {removed_count} items.")
    print("📋 Your workspace now contains:")
    
    remaining_files = [f for f in os.listdir(".") if not f.startswith(".")]
    remaining_files.sort()
    for f in remaining_files:
        if os.path.isfile(f):
            print(f"   📄 {f}")
        else:
            print(f"   📁 {f}/")

if __name__ == "__main__":
    print("=" * 60)
    print("WORKSPACE CLEANUP UTILITY")
    print("=" * 60)
    
    response = input("Remove all extra scripts? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        cleanup_files()
    else:
        print("❌ Cleanup cancelled.")
        
    print("\n" + "=" * 60)
    print("💡 USAGE GUIDE:")
    print("   To run all models: python Project_model.py")
    print("   To run Benders: uncomment the last line in Project_model.py")
    print("   To cleanup again: python cleanup_helper.py")
    print("=" * 60)