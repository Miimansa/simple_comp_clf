import argparse
from joblib import load

def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Score data using a saved model with a custom threshold.")
  parser.add_argument("model_path", type=str, help="Path to the saved model file")
  parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Custom threshold for predictions")
  parser.add_argument("-d", "--data", type=str, required=True, help="Path to the data file in CSV format")
  args = parser.parse_args()

  #Load
  model = load(args.model_path)
  #predict
  prediction = model.predict_proba(args.data)[:, 1]
  # using custom threshold and get score
  score = (prediction >= args.threshold)

  print(f"Score: {score:.4f}")

if __name__ == "__main__":
  main()

