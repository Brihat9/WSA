''' WSA ASSIGNMENT 1
    Brihat Ratna Bajracharya (19/075)
    CDCSIT
'''

import re
import nltk
import operator
import pprint # prettier print for dict and list
import copy # used to copy df dict to idf dict
import json # for storing and retrieving tf-idf values to/from file
import os # for tf-idf retrieval from file

from nltk.corpus import stopwords # for stop word Removal
from nltk.stem.porter import * # for Porter Stemmer

# total number of cranfield document to consider [1-1400]
NO_OF_FILES = 1400
NO_OF_QUERY_DOCS = 10
MAX_RELEVANT_DOCS = 10

TF_FILE = "tfs\\tf"
TFIDF_FILE = "tfidfs\\tfidf"

# TFIDF_FILE_ALL = "tfidfs\\tfidfall" # Complete TF-IDF in one file

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def header():
    ''' INFO to display on top '''
    print("\nWSA Assignment 1")
    print("Brihat Ratna Bajracharya\n19/075\nCDCSIT\n------------------------")


def cleanhtml(raw_html):
    ''' removes all HTML Tags from given string
        parameter: raw_html (string)
        result: string (text without HTML Tags)
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def tokenize(doc):
    ''' convert text to list of words in that text
        removes punctuations and whitespace, DOESNOT remove number
        parameter: doc (text)
        result: list (word list for the text)
    '''
    alphabet = re.compile('[^\w\s]')

    # remove punctuations
    clean_text = re.sub(alphabet, ' ', doc)
    # print(clean_text)

    word_list = clean_text.split()
    return word_list


def word_list_to_freq_dict(word_list):
    ''' convert list of words to dict of {word: count} pair
        parameter: word_list (list)
        result: key-value pair of word: count (dict)
    '''
    word_freq = [word_list.count(p) for p in word_list]
    return dict(list(zip(word_list,word_freq)))


def question2(word_dict_all):
    ''' QUESTION 2 of WSA Assignment 1
        sort the dict in reverse order and form list of tuple containing key and value
        prints unique word count (in all docs)
               top 10 unique words (in all docs)
               min no of unique word for half word count (in all docs)
        parameter: word_dict (dict)
        result: None
    '''

    listofTuples = sorted(word_dict_all.items() , reverse=True, key=lambda x: x[1])
    # print(listofTuples[0][0])

    unique_word_count = len(listofTuples)
    print("  Total number of CranField Document taken: " + str(NO_OF_FILES))
    print("   Number of unique words: " + str(unique_word_count))

    print('\n   Top 10 unique words:')
    for i in range(10):
        print("    " + str(i+1) + ". " + listofTuples[i][0] + ": " + str(listofTuples[i][1]))

    total_word_count = 0
    for i in range(unique_word_count):
        total_word_count += listofTuples[i][1]

    min_unique_word_for_half_count = 0
    half_word_count = 0
    for i in range(unique_word_count/2):
        min_unique_word_for_half_count += 1
        half_word_count += listofTuples[i][1]
        if(half_word_count >= total_word_count/2):
            break

    print("\n   Total number of words (T): " + str(total_word_count))
    print("   Half word count (T/2): " + str(half_word_count))
    print("   Minimum number of unique word required to reach half word count: " + str(min_unique_word_for_half_count))


def computeTF(wordDict, bagOfWords):
    ''' computes term frequency for a doc

        parameter: wordDict = {'unique_key': word_count} of one document
                   bagOfWords = word_list of one document after stop word removal
        result: dict (term_frequencies for a doc)
    '''
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents,wordDict):
    ''' computes inverse document frequency for all doc

        parameter: documents = list of {'unique_key': word_count} for all doc
                   wordDict = {'unique_key': word_count} for all doc combined
        result: dict (inverse_document_frequencies of all document)
    '''
    import math
    N = len(documents)

    ''' document frequency dict '''
    df = dict.fromkeys(wordDict.keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                df[word] += 1
    # print("Document Frequency:\n", df)

    idfDict = copy.deepcopy(df)

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    ''' computes term frequency - inverse document frequency for all doc

        parameter: tfBagOfWords = dict of term frequencies for a doc
                   idfs = dict of IDF for all document
        result: dict (tf_idf of all document)
    '''
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def get_matching_docs(tf_idf_list, query, rev_doc_num = MAX_RELEVANT_DOCS):
    ''' returns list of relevant documents for given query

        parameter: tf_idf_list = list of dict of tf-idf for all documents
                   query = query document
                   rev_doc_num = number of relevant document to return
        result: list (relevant document number for query)
    '''
    query_content_wo_tags = cleanhtml(query)
    query_tokens = tokenize(query_content_wo_tags)
    query_stemmed= [stemmer.stem(word) for word in query_tokens]
    stop_word_removed_query = [w for w in query_stemmed if not w in stop_words]

    query_weight = dict()

    for index in range(NO_OF_FILES):
        for key, val in tf_idf_list[index].items():
            for token in query_tokens:
                if token == key:
                    try:
                        query_weight[index+1] += tf_idf_list[index][key]
                    except:
                        query_weight[index+1] = tf_idf_list[index][key]

    qweight = copy.deepcopy(query_weight)
    qweight = sorted(qweight.items(), key=lambda x: x[1], reverse=True)

    relevant_docs_list = []
    rev_doc_count = 0
    for tuple in qweight:
        if rev_doc_count < rev_doc_num:
            relevant_docs_list.append(tuple[0])
            rev_doc_count += 1

    return relevant_docs_list

def main():
    ''' MAIN FUNCTION starts here '''
    header()

    ''' REQUIRED VARIABLES '''
    doc = [None] * NO_OF_FILES
    file_content = [None] * NO_OF_FILES

    ''' pre-processing '''
    content_wo_tags = [None] * NO_OF_FILES
    doc_word_list = [None] * NO_OF_FILES
    word_freq_dict_list = [None] * NO_OF_FILES
    word_freq_dict_list_2 = [None] * NO_OF_FILES

    ''' stemmer and stop words '''
    stemmed_word_list = [None] * NO_OF_FILES
    stop_word_removed_word_list = [None] * NO_OF_FILES

    ''' tf, idf, tf-idf variables '''
    tf_list = [None] * NO_OF_FILES
    tf_idf_list = [None] * NO_OF_FILES


    ''' Read CSV and create list of text from all documents read '''
    print("\nReading " + str(NO_OF_FILES) + " CranField Documents ...")
    for file_index in range(NO_OF_FILES):
        file_name = "cranfieldDocs\cranfield" +str(file_index+1).zfill(4)
        doc[file_index] = open(file_name,"r")

        if doc[file_index].mode == 'r':
            print("  Reading doc: " + file_name + "\r"),
            file_content[file_index] = doc[file_index].read()

        doc[file_index].close()
    print("\n  " + str(NO_OF_FILES) + " documents read\n")

    ''' print content of one doc (testing) '''
    # print(file_content[1])


    ''' Removes all HTML Tags and create word list (separate for each doc) '''
    print("Cleaning HTML tags and tokenizing ..."),
    for index in range(NO_OF_FILES):
        content_wo_tags[index] = cleanhtml(file_content[index])
        doc_word_list[index] = tokenize(content_wo_tags[index])
    print("DONE.\n")


    ''' Porter Stemmer '''
    ''' made global '''
    # stemmer = PorterStemmer()

    print("Stemming (Porter Stemmer) ..."),
    for index in range(NO_OF_FILES):
        # [stemmer.stem(word) for word in doc_word_list[index]]
        stemmed_word_list[index] = [stemmer.stem(word) for word in doc_word_list[index]]
    print("DONE.\n")


    ''' Stop Word Removal '''
    ''' made global '''
    # stop_words = set(stopwords.words('english'))

    print("Removing stop words ..."),
    for index in range(NO_OF_FILES):
        stop_word_removed_word_list[index] = [w for w in stemmed_word_list[index] if not w in stop_words]
    print("DONE.\n")

    ''' Test for Stemmer and Stop Word Removal '''
    # print("word list [0] vs after stemming and stop word removal [0]")
    # print(len(stemmed_word_list[0]))
    # print(len(stop_word_removed_word_list[0]))


    ''' word_list -> {word: freq} pair for each document -> list of these dict for all document '''
    ''' word_list_2 without stop words -> {word: freq} pair for each document -> list of these dict for all document '''
    print("Creating LIST of each word, word_count pair ..."),
    for index in range(NO_OF_FILES):
        word_freq_dict_list[index] = word_list_to_freq_dict(doc_word_list[index])
        word_freq_dict_list_2[index] = word_list_to_freq_dict(stop_word_removed_word_list[index])
    print("DONE.\n")


    ''' word_dict_all -> creates dict of all unique words (all doc combined) {unique_word: word_count} pair '''
    ''' word_dict_all_2 -> creates dict of all unique words (all doc combined) {unique_word: word_count} pair (no stop words) '''
    word_dict_all = {}
    word_dict_all_2 = {}

    print("Creating DICT of (unique_word, word_count) pair ..."),
    for index in range(NO_OF_FILES):
        for key in word_freq_dict_list[index]:
            word_dict_all[key] = word_freq_dict_list[index][key]
        for key in word_freq_dict_list_2[index]:
            word_dict_all_2[key] = word_freq_dict_list_2[index][key]
    print("DONE.\n")

    ''' Test for unique word count, with and without stop words '''
    # print("unique word count before after stemming")
    # print(len(word_dict_all.keys()))
    # print(len(word_dict_all_2.keys()))


    ''' Question 2 of WSA 1 Assigment '''
    print("Solution to Q2 of WSA 1\n")

    print(" With Stop Words (Question 2)\n ----------------------------")
    question2(word_dict_all)

    print("\n\n Without Stop Words (Question 3)\n -------------------------------")
    question2(word_dict_all_2)


    path, dirs, files = next(os.walk("tfidfs"))
    file_count = len(files)

    if file_count >= NO_OF_FILES:
        print("\nPre-computed TF-IDF values found")
        print(" Reading pre-computed TF-IDF values ..."),
        for index in range(NO_OF_FILES):
            tfidf_file = open("tfidfs\\" + files[index], 'r')
            tfidf_dict_res = json.loads(tfidf_file.readline())
            tf_idf_list[index] = tfidf_dict_res
        print("DONE.")

        ''' READING COMPLETE TF-IDF IN ONE FILE '''
        # tfidf_file_all = open(TFIDF_FILE_ALL, 'r')
        # tf_idf_list = json.loads(tfidf_file_all.readline())
        # tfidf_file_all.close()
    else:
        ''' TF, IDF, TF-IDF part here '''
        ''' calculate term frequencies for each document '''
        print("\nPre-computed TF-IDF values not found")
        print("\nCalculating Term Frequencies ..."),
        for index in range(NO_OF_FILES):
            tf_file = open(TF_FILE + str(index).zfill(4), "w+")
            tf_list[index] = computeTF(word_freq_dict_list_2[index], stop_word_removed_word_list[index])
            tf_file.write(str(tf_list[index]))
            tf_file.close()
        print("DONE.\n")

        ''' calculate inverse document frequencies of all document '''
        print("Calculating Inverse Document Frequencies ..."),
        idf = computeIDF(word_freq_dict_list_2,word_dict_all_2)
        print("DONE.\n")

        ''' calculates TF-IDF for all document '''
        print("Calculating TF-IDF ..."),
        for index in range(NO_OF_FILES):
            tfidf_file = open(TFIDF_FILE + str(index).zfill(4), "w+")
            tf_idf_list[index] = computeTFIDF(tf_list[index], idf)
            json.dump(tf_idf_list[index], tfidf_file)
            tfidf_file.close()

        ''' Storing TF-IDF to single file '''
        # tfidf_file_all = open(TFIDF_FILE_ALL, "w+")
        # json.dump(tf_idf_list, tfidf_file_all)
        # tfidf_file_all.close()

        print("DONE.\n")


        ''' Display TF, IDF and TF-IDF values '''
        # print(tf_list)
        # print(idf)
        # print(tf_idf_list)

        ''' Pretty display TF-IDF '''
        # pprint.pprint(tf_idf_list)

    ''' MATCHING AND RANKING PART HERE '''
    query_docfile = open("query_documents", 'r')
    query_docs = [None] * NO_OF_QUERY_DOCS

    ''' reading query documents '''
    for index in range(NO_OF_QUERY_DOCS):
        query_docs[index] = query_docfile.readline()

    ''' for all QUERY DOCS '''
    relevant_docs_res = [None] * NO_OF_QUERY_DOCS

    ''' Finding relevant documents for queries '''
    print("\nFinding relevant documents for queries ...")

    for index in range(NO_OF_QUERY_DOCS):
        print(" Matching relevant documents for Query Doc " + str(index+1) + " ..."),
        relevant_docs_res[index] = get_matching_docs(tf_idf_list, query_docs[index])
        print("DONE.")

    print("DONE.")

    print("\n\nRESULT: RELEVANT DOCUMENT FOR QUERIES")
    print("-------------------------------------")
    for index in range(NO_OF_QUERY_DOCS):
        print("\nMatched relevant documents for Query Doc  " + str(index+1))
        print(relevant_docs_res[index])

    print("\nDONE.")


if __name__ == "__main__":
     main()
     # raw_input()
