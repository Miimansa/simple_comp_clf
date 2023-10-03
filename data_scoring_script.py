import argparse
from argparse import RawTextHelpFormatter
from joblib import load
import os

import nltk
# Step 1: word_tokenize
from nltk.tokenize import word_tokenize
if not nltk.data.find("tokenizers/punkt"):
    nltk.download("punkt")
# Step 2: stop_words removal
from nltk.corpus import stopwords
if not nltk.data.find("corpora/stopwords"):
    nltk.download("stopwords")
stopwords = stopwords.words('english')
#Step 3: stemming
from nltk.stem import SnowballStemmer

def text_preprocessor(text):
    return [SnowballStemmer(language='english').stem(word) for word in word_tokenize(text) if word not in stopwords]

def main():

    model_help_text = """Model to be used 
    v1 - ensembled(logit,svm,rf,gb)
    v2 - """

    threshold_help_text = """Custom threshold for predictions where precision of simple class is 0.99
    For v1, t1 = 0.26  """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Score data using a saved model with a custom threshold.",formatter_class=RawTextHelpFormatter)
    parser.add_argument("model", type=str, help=model_help_text)
    parser.add_argument("-t", "--threshold",  type=float, default=0.5, help=threshold_help_text)
    parser.add_argument("-d", "--data", type=str, required=True, help="Data")
    args = parser.parse_args()
    
    #Load model
    if args.model == "v1":
        model = load(os.getcwd()+"/models/"+"sc_v1.joblib")
    else:
        print(args.model+" not configured yet")
    
    # Load the TF-IDF vectorizer
    vectorizer = load(os.getcwd()+"/vectorizers/tfidf_vectorizer.joblib")

    data = vectorizer.transform([args.data])

    #predict
    prediction_1 = model.predict_proba(data)[:, 1]
    prediction_0 = model.predict_proba(data)[:, 0]
    # using custom threshold get score
    score = prediction_1 >= args.threshold
    
    print(f"Simple: {not score[0]}")
    print(f"Simple_Prob_Score: {prediction_0[0]}")

if __name__ == "__main__":
    main()

