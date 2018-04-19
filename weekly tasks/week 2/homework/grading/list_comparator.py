'''
A script to compare the word lists created for programming assignment 2.

The imported file code is the file that the student created.
'''

import sys
import code

def main():
    
    gold_words = set()
    student_words = set()

    try:
        with open("1000frequentwords.txt", 'rt', encoding="utf-8") as goldFile:
            for line in goldFile:
                elements = line.split()
                if elements:
                    gold_words.add(elements[0])

        freq_dict = code.collect_frequencies(code.textFile)
                
    except FileNotFoundError:
        print("One of the files does not exist on your computer.")
        sys.exit(0)

    else:
        student_words, freqs = code.find_frequent_words(freq_dict, 1000)
        
    if len(gold_words) != len(student_words):
        print ('The lists are of different size. Please make sure to only ' +
        'use equally sized lists.')
        sys.exit(0)
        
    print("Words that appear in the gold list and not in the student list:")
    print(gold_words.difference(student_words))
    print("Words that appear in the student list and not in the gold list:")
    print(set(student_words).difference(gold_words))
    
    intersection = gold_words.intersection(student_words)
    overlap = float(len(intersection))/len(gold_words)*100

    print('The overlap between the gold list and the student output is {}%'.format(overlap))

    temp_test = []
    for i in range(len(student_words)-1):
        if freqs[i] == freqs[i+1]:
            temp_test.append(student_words[i])
        else:
            if temp_test:
                temp_test.append(student_words[i])
                if temp_test != sorted(temp_test):
                    print('The words with the same freq are not ordered alphabetically, cf. {}'.format(temp_test))
                    break
            temp_test = []

if __name__ == '__main__':
    main()
