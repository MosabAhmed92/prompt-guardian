import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset # huggingface dataset

load_dotenv()

seed = int(os.getenv("SEED"))
dataset_path = os.getenv("DATASET_PATH")

data = load_dataset(path = dataset_path, name = "core")

# defining the traingin and testing sets

train_df = data['train'].to_pandas()
test_df = data['test'].to_pandas()
validation_df = data['validation'].to_pandas()

#dropping NAN and missing texts on both train and test sets 

print("Dropping Nan and empty text enteries in the dataset ... ")
train_df = train_df.dropna(subset='text')
test_df = test_df.dropna(subset='text')
validation_df = validation_df.dropna(subset = 'text')

print(f"Shape of train {train_df.shape}")
print(f"Shape of test {test_df.shape}")
print(f"Shape of validation {validation_df.shape}")


# leakage check --> making sure no text from train is on the test 
if test_df['text'].isin(train_df['text']).sum() > 0:
    print("Duplication Found ... cleaning ..")
    test_df = test_df[~test_df['text'].isin(train_df['text'])]
else:
    print("No Text duplication Found between testing set and training set ...")

if validation_df['text'].isin(train_df['text']).sum() > 0:
    print("Duplication Found ... cleaning ..")
    validation_df = validation_df[~validation_df['text'].isin(train_df['text'])]
else:
    print("No Text duplication Found between vaildation set and training set ...")


print("Using jupyter it has been confirmed that all the categories named : benign and edge_cases are labeled with 0 --> harmless prompts")

# Checking if the data is well balanced (we will work on the label class and forget about the category class)


#constructing a full dataframe to test the label balance on the wholedataset
full_df = pd.concat([train_df, test_df, validation_df], axis = 0, ignore_index=True)

percentage = full_df['label'].value_counts()/len(full_df)


# label 0 ----> Benign Sample
# label 1 ----> Malicious Sample
# bar chart in EDA.ipynb
for key,value in percentage.items():
    print(f"the items labeled {key} are {value}% of the whole dataset")

# The average character length for benign texts and the average character length for malicious texts
# Huersitic Bias

train_df['length_of_text'] = train_df['text'].str.len()
test_df['length_of_text'] = test_df['text'].str.len()

print(f"The average length of text (for Benign Samples) in Training dataset is {train_df[train_df['label'] == 0]['length_of_text'].mean()}")
print(f"The average length of text (for Malicious Samples) in Training dataset is {train_df[train_df['label'] == 1]['length_of_text'].mean()}")


print(f"The average length of text (for Benign Samples) in Testing dataset is {test_df[test_df['label'] == 0]['length_of_text'].mean()}")
print(f"The average length of text (for Malicious Samples) in Testing dataset is {test_df[test_df['label'] == 1]['length_of_text'].mean()}")


# def calculate_everage(df):
#     #calculate the average number of texts in dataframe where label = 0 and where label = 1
#     length = 0
#     for i, text in enumerate(df['text']):
#         length_of_text = len(str(text))
#         length += length_of_text
#     #calculate the average length of text
#     average = length/len(df)
#     print(f"The average length of text in the dataframe is {average}")
#     return average

# calculate_everage(train_df)
# calculate_everage(test_df)

print("Saving the cleaned dataset ... ")

train_df.to_csv("train_clean.csv", index = False)
test_df.to_csv("test_clean.csv", index=False)
validation_df.to_csv("validation_clean.csv", index = False)

print("----- Saved -----")

