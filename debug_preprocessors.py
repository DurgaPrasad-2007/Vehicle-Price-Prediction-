"""
Debug script to check preprocessors
"""

from src.data.preprocessing import get_preprocessor

def debug_preprocessors():
    """Debug preprocessor state"""
    preprocessor = get_preprocessor()
    
    print("=== Preprocessor Debug ===")
    print(f"Label encoders: {list(preprocessor.label_encoders.keys())}")
    print(f"Scalers: {list(preprocessor.scalers.keys())}")
    print(f"Feature columns: {len(preprocessor.feature_columns)}")
    
    # Try to load preprocessors
    print("\n=== Loading Preprocessors ===")
    preprocessor.load_preprocessors()
    
    print(f"After loading - Label encoders: {list(preprocessor.label_encoders.keys())}")
    print(f"After loading - Scalers: {list(preprocessor.scalers.keys())}")
    print(f"After loading - Feature columns: {len(preprocessor.feature_columns)}")

if __name__ == "__main__":
    debug_preprocessors()
