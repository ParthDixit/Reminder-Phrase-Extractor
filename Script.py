import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfTransformer


##Set the paths of training data and test data

train_path = 'training_data.tsv'
test_path = 'eval_data.txt'

## Intializing the classifier trainer
print("Training Classifier...")
import Classifier
Classifier.classifier_trainer_main(train_path)
print("Training Completed.")

##Intiate Pattern Generator

print("Intilizing Extraction Process....\nPlease Wait for a while.")

# import  REGEX_Gen
# REGEX_Gen.regex_gen_main(train_path)

import Pattern_Generator
Pattern_Generator.pattern_gen_main(train_path)

## Loading test data for classification

file = open(test_path, 'r')
data = []
for word in file:
    data.append(' '.join(word.split()))

data_pd = pd.DataFrame(data)
data_pd = data_pd[0]  # Setting the correct dimensions

## Loading Vectorizer & Classifier(Logistic Regression)

f = open("vector.pickle",'rb')
vector = pickle.load(f)  # Vectorizer Loaded
f.close()

f = open("cls.pickle",'rb')
cls = pickle.load(f)  # Classifier Loaded
f.close()

print("Vocabulary & classifier pickle files are loaded by My_Script.")
print("It uses them to predict if a sentence from eval_data.txt contains a reminder or not.")

x_test = vector.transform(data)  # Vocabulary is Generated for test data
np.asarray(x_test)  # Converted to array

x_test_tfidf = TfidfTransformer().fit_transform(x_test)  # Repeating words like 'the' is reduced

prediction = cls.predict(x_test_tfidf)  # Predications are derived
print("Predictions Made.")

output = []
for i in prediction:
    output.append(i)

output_list = list(output)  # This lists saves the predictions

## Saving Bag Of Words Model Output

BOW = pd.DataFrame(data={'sent': data_pd, 'label': output_list})
BOW.to_csv("Bag of Words.csv", index=False)
print("BOW model results have been saved as Bag of Words.csv .")

##Intiate Regex Matcher

import Pattern_Matcher
submission = Pattern_Matcher.extraction_main(BOW)
print("Extraction Completed.")

## Creating Final Result

print("The sentences which classifier predicted to have reminder are replaced by extracted phrases.")
submission.to_csv('Output.csv', index=False)
print("Final Result have been saved as Output.csv .")