import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

load_dotenv()
seed = int(os.getenv("SEED"))

# loading the cleaned datasets 
train_df = pd.read_csv("train_clean.csv")
validation_df = pd.read_csv("validation_clean.csv")
test_df = pd.read_csv("test_clean.csv")

print("Clean datasets Loaded successfully ... ")

# define feature and target varaibles
X_train = train_df['text']
y_train = train_df['label']

# remember to keep this test aside and never show it to the model untill we select the model and the hyperparameters for the model. This is to avoid data leakage and overfitting.
X_test = test_df['text']
y_test = test_df['label']

X_val = validation_df['text']
y_val = validation_df['label']

# # this is our selected model from the experiments.py file.
# exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(random_state=seed, max_iter=1000))
# exp_b_pipe.fit(X_train, y_train)
# y_pred_exp_b = exp_b_pipe.predict(X_val)
# print("classification Report for untuned logisitic regression - Exp.B applying N-gram (1, 2)")
# print(classification_report(y_true=y_val, y_pred=y_pred_exp_b))

# we will go further with exp_b_pipe (selected and locked model)
# 1- Vocabulary size control using max_features : Test whether limiting TF-IDF vocabulary improves validation performance and reduces false positives.

# max_features_options = [3000,5000,10000,None]

# for max_feature in max_features_options:
#     sweep_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words=None, max_features=max_feature),
#                                LogisticRegression(random_state=seed, max_iter=1000))
#     sweep_pipe.fit(X_train, y_train)
#     y_pred_sweep = sweep_pipe.predict(X_val)
#     print(f"Classification Report for our sweep pipleline when max_feautres is {max_feature} is :")
#     print(classification_report(y_true = y_val, y_pred = y_pred_sweep))

# max_features limitation got rejected.
# Keep max_features=None.

# 2- 
# min_df_list = [1, 2, 3, 4, 5]
# for min_df in min_df_list:
#     exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=min_df), LogisticRegression(random_state=seed, max_iter=1000))
#     exp_b_pipe.fit(X_train, y_train)
#     y_pred_exp_b = exp_b_pipe.predict(X_val)
#     print(f"Classification Report for our chosen pipeline when min_df is {min_df} is :")
#     print(classification_report(y_true = y_val, y_pred = y_pred_exp_b))

# we will go with min_df = 1 

# 3 - 

# max_df_list = [0.70, 0.90, 0.95, 1.0]

# for max_df in max_df_list:
#     exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=max_df), LogisticRegression(random_state=seed, max_iter=1000))
#     exp_b_pipe.fit(X_train, y_train)
#     y_pred_exp_b = exp_b_pipe.predict(X_val)
#     print(f"Classification Report for our chosen pipeline when minmax_df is {max_df} is :")
#     print(classification_report(y_true = y_val, y_pred = y_pred_exp_b))

# we will with max_df = 1 

# 4-
# working on th elogistic regression hyper parameter 

# c_list = [0.01, 0.1, 1.0, 3.0, 10.0]

# for c in c_list:
#     exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=1.0), LogisticRegression(random_state=seed, max_iter=1000, C = c))
#     exp_b_pipe.fit(X_train, y_train)
#     y_pred_exp_b = exp_b_pipe.predict(X_val)
#     print(f"Classification Report for our chosen pipeline when different is {c} is :")
#     print(classification_report(y_true = y_val, y_pred = y_pred_exp_b))

# 5- 
# we will go either with c = 1.0 or c = 10.0 

# c_list = [1.0, 10.0]
# class_weights = [None, "balanced"]
# for class_weight in class_weights:
#     for c in c_list:
#         exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=1.0), LogisticRegression(random_state=seed, max_iter=1000, C = c, class_weight=class_weight))
#         exp_b_pipe.fit(X_train, y_train)
#         y_pred_exp_b = exp_b_pipe.predict(X_val)
#         print(f"Classification Report for our chosen pipeline when class weight is {class_weight} and c is {c} :")
#         print(classification_report(y_true = y_val, y_pred = y_pred_exp_b))


# 6 - the threshold tuning 

# c_list = [1.0, 10.0]
# thresholds = [
#     0.30,
#     0.35,
#     0.40,
#     0.45,
#     0.50,
#     0.55,
#     0.60,
#     0.65,
#     0.70
# ]
# for c in c_list:
#     exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=1.0),
#                             LogisticRegression(random_state=seed, max_iter=1000, C = c, class_weight=None))
#     exp_b_pipe.fit(X_train, y_train)
#     y_pred_exp_b = exp_b_pipe.predict_proba(X_val)
#     malicious_probabilities = y_pred_exp_b[:, 1]
#     print("=" * 80)
#     print(f"Threshold tuning results for Logistic Regression with C={c}")
#     print("=" * 80)    
#     for threshold in thresholds:
#         y_pred_threshold = (malicious_probabilities >= threshold).astype(int)
#         print(f"\nC={c} | threshold={threshold}")
#         print(classification_report(y_true=y_val, y_pred=y_pred_threshold))


# Selected model: C=10.0 and threshold=0.35

selected_classifier = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=1.0),
                            LogisticRegression(random_state=seed, max_iter=1000, C = 10.0, class_weight=None))

selected_classifier.fit(X_train, y_train)
y_pred = selected_classifier.predict_proba(X_val)
y_proba = y_pred[:, 1]
threshold = 0.35
y_proba_threshold = (y_proba >= threshold).astype(int)
# probability >= 0.35 → malicious
# probability < 0.35  → safe

print("Classification Report :")
print(classification_report(y_true = y_val, y_pred=y_proba_threshold))

print(confusion_matrix(y_true = y_val, y_pred = y_proba_threshold))

# creating validation analysis dataframe : 7 malicious prompts got passed (FN = 7) as per the last confusion matrix

# construct the dataframe theat has those columns : text, label, predicted_label, probability, error_type only for the misclassified prompts (FN and FP)
false_indices = np.where(y_val != y_proba_threshold)[0] # list of indices where the results are wrong FP and FN

validation_error_df = validation_df.iloc[false_indices].copy()
validation_error_df["original_index"] = false_indices
validation_error_df["true_label"] = y_val.iloc[false_indices].values
validation_error_df["predicted_label"] = y_proba_threshold[false_indices]
validation_error_df['malicious_probability'] = y_proba[false_indices]

validation_error_df['error_type'] = np.where((validation_error_df["true_label"] == 0)&(validation_error_df['predicted_label'] == 1), "FP", "FN")

print(validation_error_df['error_type'].value_counts())

validation_error_df.to_csv("validation_error.csv", index = False)

# FN csv 
validation_error_df[validation_error_df['error_type'] == "FN"].to_csv("validation_error_FN.csv", index=False)
#FP csv
validation_error_df[validation_error_df['error_type'] == "FP"].to_csv("validation_error_FP.csv", index=False)


# finally testing on test set --> model selceted is :
# selected_classifier = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=None, max_df=1.0),
                            #LogisticRegression(random_state=seed, max_iter=1000, C = 10.0, class_weight=None))

# final Test evaluation 

y_pred_proba_test = selected_classifier.predict_proba(X_test)[:, 1]
y_pred_proba_threshold = (y_pred_proba_test >= threshold).astype(int)

print("Classification Report on Test - set:")
print(classification_report(y_true = y_test, y_pred=y_pred_proba_threshold))

print(confusion_matrix(y_true = y_test, y_pred = y_pred_proba_threshold))

# saving the model : 
joblib.dump(selected_classifier,"selected_classifier.joblib")