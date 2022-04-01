import numpy as np
import pandas as pd
import re
from collections import Counter

def regex_gen_main(path):

    ## Slicing Data as Sent and Label

    data = pd.read_csv(path, header=0, delimiter='\t', quoting=3)
    list_sent = data['sent']
    list_label = data['label']

    clean_list = []
    for i,label in enumerate(list_label):
        if label == 'Not Found':
            pass
        else:
            clean_list.append([list_sent[i], label])

    ## Extracting pattern with Label at the Middle

    some_list = []
    for i in clean_list:
        label = i[1]
        label = re.sub('[^A-Za-z0-9]+', ' ', label)
        pat = '(\w*)\W*('+label+')\W*(\w*)'
        sent = i[0]
        for i in re.findall(pat, sent, re.I):
            some_list.append([" ".join([x for x in i if x != '']), label])

    words_list = []
    for word in some_list:
        if word[1] in word[0]:
            words_list.append(word[0].replace(word[1], ' '))

    pattern_list = Counter(words_list).most_common()
    # print(pattern_list)
    pattern_list1 = [tp[0] for tp in pattern_list]
    # print(pattern_list1)

    # Extracting pattern with Label at the Right

    some_list = []
    for i in clean_list:
        label = i[1]
        label = re.sub('[^A-Za-z0-9]+', ' ', label)
        pat = '(\w*)\W*(\w*)\W*(' + label + ')'
        sent = i[0]
        for i in re.findall(pat, sent, re.I):
            some_list.append([" ".join([x for x in i if x != '']), label])

    words_list = []
    for word in some_list:
        if word[1] in word[0]:
            words_list.append(word[0].replace(word[1], ' '))
    pattern_list = Counter(words_list).most_common()
    pattern_list2 = [tp[0] for tp in pattern_list]

    ## Saving the Pattern lists for Regex Matcher

    import pickle
    f = open("pattern1.pickle", 'wb')
    pickle.dump(pattern_list1,f)
    f.close()

    f = open("pattern2.pickle", 'wb')
    pickle.dump(pattern_list2,f)
    f.close()








