from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
import pickle


def classifier_trainer_main(path):
    ## Loading & slicing the training data for classification.
    print("training_data.tsv is used to train the ML classifier")
    train_path = 'training_data.tsv'
    data = pd.read_csv(path, header=0, delimiter='\t', quoting=3)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.loc[data['label'] != 'Not Found', 'label'] = "Found"  ## Where the label is a phrase, it is turned to 'Found'
    x_train = (data['sent'])
    y_train = (data['label'])

    clean_sent = []  # List contains all the 'sent' data,
    for i in x_train:
        clean_sent.append(i)

    ## Vocabulary Generation

    vector = CountVectorizer(ngram_range=(1, 2))
    clean_vector = vector.fit_transform(clean_sent)  # Vocabulary is generated for training data .
    np.asarray(clean_vector)  # Converted into array for Tf-IDF.

    tfidf = TfidfTransformer()
    x_train_tfidf = tfidf.fit_transform(clean_vector)

    ## Choose classifier.
    
    # cls = MLPClassifier(activation='logistic', max_iter=10, hidden_layer_sizes=[100]) 
    cls = LogisticRegression()
    cls.fit(x_train_tfidf, y_train)  # Logistic Regression Classifier is trained.
    y_test = cls.predict(x_train_tfidf)

    print('Model Accuracy Score: %.2f' % (accuracy_score(y_train, y_test)*100))

    ## Saving Vectorizer & Classifier

    f = open("vector.pickle", 'wb')
    pickle.dump(vector, f)
    f.close()

    f = open("cls.pickle", 'wb')
    pickle.dump(cls, f)
    f.close()

    print("Vocabulary & Classifier are saved as pickle file by Classifier.")

# classifier_trainer_main('training_data.tsv')
