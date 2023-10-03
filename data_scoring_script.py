import argparse
from joblib import load
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Score data using a saved model with a custom threshold.")
    parser.add_argument("model", type=str, help="Model to be used")
    parser.add_argument("-t", "--threshold",  type=float, default=0.5, help="Custom threshold for predictions")
    parser.add_argument("-d", "--data", type=str, required=True, help="Data")
    args = parser.parse_args()
    
    #Load model
    if args.model == "v1":
        model = load(os.getcwd()+"/models/"+"sc_v1.joblib")
    else:
        print(args.model+" not configured yet")
    
    # Load the TF-IDF vectorizer
    vectorizer = load(os.getcwd()+"/vectorizers/tfidf_vectorizer.joblib")
    data = vectorizer.transform(args.data)

    #predict
    prediction = model.predict_proba(data)[:, 1]
    # using custom threshold get score
    score = (prediction >= args.threshold)
    
    print(f"Score: {score:.4f}")

if __name__ == "__main__":
    main()

