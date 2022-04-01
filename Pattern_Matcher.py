import pandas as pd
import numpy as np
import re
import pickle

# def extraction_main(path):
def extraction_main(BOW):
    ## Loading Pattern lists.

    # data = open(path,'r')

    # data = pd.DataFrame(BOW, header=0)

    f = open('pattern1.pickle',"rb")
    pat1= pickle.load(f)
    f.close()

    f = open('pattern2.pickle',"rb")
    pat2= pickle.load(f)
    f.close()

    print("Pattern pickle files are loaded by Pattern_Matcher.")
    print("Pattern_Matcher will use these pickle files to extract possible reminder phrases from eval_data.txt.")

    ## Recognizing Patterns for Phrase Extraction
    # print(BOW)

    sent = BOW['sent']
    label = BOW['label']

    for idx in range(0, len(BOW)):

        if label[idx] == 'Not Found':
            continue

        small_list = []  # This list will contain the all possible extractions from the text.
        found = 0

        for pat in pat1:  # Pattern matching with label at center.
            boundary = pat.split()
            if len(boundary) == 2:

                m = re.search(boundary[0] + " " '(.+?)'  " "+ boundary[1], sent[idx])
                if m:
                    found = m.group(1)
                    small_list.append(found)
                else:
                    pass

        for pat in pat2:  # Pattern matching with label at right.
            boundary = pat.split()
            if len(boundary) == 2:

                m = re.search(boundary[0] + " " + boundary[1] + " " '(.*)', sent[idx])
                if m:
                    found = m.group(1)
                    # print(found)
                    small_list.append(found)
                else:
                    continue

        if found == 0:  # If no pattern is matched then the result is "Not Found".
            label[idx] = 'Phrase Not Found'
            continue

        if len(small_list) == 1:
            label[idx] = small_list[0]

        else:
            type_score = []
            for type in small_list:
                words = type.split()
                Penalized_list = ['on', 'in', 'to', 'the']  # These words increases the score unnecessarily.
                score = len(words)
                for word in words:
                    if word in Penalized_list:  # if the word belongs to Penalized list the score will be reduced
                        score = score - 3
                    else:
                        pass
                type_score.append(score)
            m = max(type_score)  # Max score is selected
            ind = [i for i, j in enumerate(type_score) if j == m]  # Index values of all the extractions with max scores are saved
            index = ind[-1]  ## The last extraction is selected.
            label[idx] = small_list[index]  ## The extraction is replaced.

    return BOW
