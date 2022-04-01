import nltk
import pandas as pd
from collections import Counter

nltk.download('punkt')
from nltk import word_tokenize

def pattern_gen_main(path):
    data = pd.read_csv(path, header=0, delimiter='\t', quoting=3)
    print("training_data.tsv is used to extract different patterns which helps in identifying reminder.")
    x_sent = data['sent']
    y_label = data['label']
    clean_list = []
    for i, label in enumerate(y_label):
        if label != "Not Found":
            clean_list.append([x_sent[i], label])
    # print(clean_list)

    master_list1 = []
    master_list2 = []

    for row in clean_list:
        sent = row[0]
        label = row[1]
        token_sent = word_tokenize(sent)
        token_label = word_tokenize(label)
        no_pattern = [",", '.', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', ':', ';',
                      '/']
        # print(token_sent,token_label)
        clean_sent = [token for token in token_sent if token not in no_pattern]
        # print(clean_sent,token_label,token_label[-1])
        pattern = []

        for idx, w in enumerate(clean_sent):
            if token_label[0] == clean_sent[idx]:
                if idx == 0:
                    break
                else:
                    pattern.append(clean_sent[idx - 1])
                    break

            if token_label[-1] == clean_sent[idx]:
                if int(idx) == int(len(clean_sent) - 1):
                    break
                else:
                    pattern.append(clean_sent[idx + 1])
                    break

        # for idx, w in enumerate(clean_sent):
        #     if token_label[-1] == clean_sent[idx]:
        #
        #         if int(idx) == int(len(clean_sent) - 1):
        #             break
        #         else:
        #             pattern.append(clean_sent[idx + 1])
        #             break

        master_list1.append(pattern)
        pattern2 = []

        for idx, w in enumerate(clean_sent):

            if token_label[0] == clean_sent[idx]:
                if idx - 2 < 0:
                    break
                else:
                    pattern2.append(clean_sent[idx - 2])
                    pattern2.append(clean_sent[idx - 1])
                    break

        master_list2.append(pattern2)

    # print(master_list)
    # print(10*'\n')
    temp_list1 = []
    for i in master_list1:
        if len(i) == 2:
            temp_list1.append(" ".join([x for x in i]))
    pattern_list = Counter(temp_list1).most_common()
    # print(pattern_list)
    pattern_list1 = [i[0] for i in pattern_list]
    # print(pattern_list1)

    temp_list2 = []
    for i in master_list2:
        if len(i) == 2:
            temp_list2.append(" ".join([x for x in i]))
    pattern_list_2 = Counter(temp_list2).most_common()
    pattern_list2 = [i[0] for i in pattern_list_2]

    import pickle
    f = open("pattern1.pickle", 'wb')
    pickle.dump(pattern_list1, f)
    f.close()

    f = open("pattern2.pickle", 'wb')
    pickle.dump(pattern_list2, f)
    f.close()
    print("Patterns are Extracted and saved as pickle file by Pattern_Generator")
