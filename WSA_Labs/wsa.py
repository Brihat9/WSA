''' WSA ASSIGNMENT 1
    Brihat Ratna Bajracharya (19/075)
    CDCSIT
'''

import re
import nltk
import math # for log function
import pprint # prettier print for dict and list
import copy # used to copy df dict to idf dict
import json # for storing and retrieving tf-idf values to/from file
import os # for tf-idf retrieval from file
import numpy as np # to calculate cosine similarity

from collections import Counter # for unique count
from nltk.corpus import stopwords # for stop word Removal
from nltk.stem.porter import * # for Porter Stemmer
from numpy import asarray, savetxt, loadtxt # for storing and loading document vectors

# total number of cranfield document to consider [1-1400]
NO_OF_FILES = 1400
NO_OF_QUERY_DOCS = 10 # total number of query documents [1-10]
MAX_RELEVANT_DOCS = 10 # default number of relevant document to return

folders = ["tfs", "tfidfs", "docvec", "precision_recall_result"]

# create folders if they do not exist
for folder_name in folders:
    if not os.path.exists(folder_name):
        print("Creating directory: " + folder_name + " ..."),
        os.mkdir(folder_name)
        print("DONE.")

TF_FILE = "tfs\\tf"
TFIDF_FILE = "tfidfs\\tfidf"

DOC_VEC_FILE = "docvec\\docvecall.csv"
DOC_VEC_LENGTH_FILE = "docvec\\docveclength"

RELEVANT_DOCS_LIST_FILE = "relevant_doc_ids"
PRECISION_RECALL_DIR = "precision_recall_result\\precision_recall_"

''' some global variables defined for cosine similarity calculation '''
UNIQUE_WORD_LIST = []
UNIQUE_WORD_COUNT = 0
DOCUMENT_FREQUENCY = {}
TF_IDF_LIST = []

# TFIDF_FILE_ALL = "tfidfs\\tfidfall" # Complete TF-IDF in one file

stemmer = PorterStemmer()

''' if stopwords not found, download it from nltk '''
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
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


def computeDF(documents, wordDict):
    ''' calculates and returns document frequency for all documents

        parameter: documents = list of {'unique_key': word_count} for all doc
                   wordDict = {'unique_key': word_count} for all doc combined

        result: dict of {word: document_frequency} for all document
    '''
    df = dict.fromkeys(wordDict.keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                df[word] += 1
    return df


def computeIDF(documents,wordDict):
    ''' computes inverse document frequency for all doc

        parameter: documents = list of {'unique_key': word_count} for all doc
                   wordDict = {'unique_key': word_count} for all doc combined
        result: dict (inverse_document_frequencies of all document)
    '''
    import math
    N = len(documents)

    ''' document frequency dict '''
    ''' changed: now calls separate function '''
    df = computeDF(documents, wordDict)

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


def create_document_vector(word_dict_list):
    ''' creates document vector for all documents '''

    ''' GLOBAL VARIABLES '''
    global NO_OF_FILES
    global UNIQUE_WORD_LIST
    global UNIQUE_WORD_COUNT
    global TF_IDF_LIST

    # initialize num_of_doc x total_unique_word_count matrix
    document_vectors = np.zeros((NO_OF_FILES, UNIQUE_WORD_COUNT))

    for index in range(NO_OF_FILES):
        word_list = word_dict_list[index].keys()
        for word in word_list:
            try:
                word_index = UNIQUE_WORD_LIST.index(word)
                document_vectors[index][word_index] = TF_IDF_LIST[index][word]
            except:
                pass

    ''' testing purpose, shows non zero element of document_vectors '''
    # for index in range(NO_OF_FILES):
    #     for index2 in range(len(unique_word_list)):
    #         if document_vectors[index][index2] != 0.0:
    #             print(index, index2, document_vectors[index][index2])

    return document_vectors


def preprocess_query(query):
    ''' functions that preprocesses given query text and
        returns list of tokens
    '''

    # removes html tags (if any)
    query_content_wo_tags = cleanhtml(query)

    # tokenize given query doc
    query_tokens = tokenize(query_content_wo_tags)

    # applies Porter Stemmer to tokens
    query_stemmed= [stemmer.stem(word) for word in query_tokens]

    # removes stop words from stemmed tokens
    stop_word_removed_query = [w for w in query_stemmed if not w in stop_words]

    # returns processed tokens for given query
    return stop_word_removed_query


def generate_document_vector(query_doc):
    ''' generates document vector given query '''

    ''' GLOBAL VARIABLES '''
    global NO_OF_FILES
    global UNIQUE_WORD_LIST
    global UNIQUE_WORD_COUNT
    global DOCUMENT_FREQUENCY

    # first preprocess the query
    processed_query_tokens = preprocess_query(query_doc)

    # initialize query vector, default value 0
    query_vector = np.zeros(UNIQUE_WORD_COUNT)
    # print("Init query_vector (all zeros): ", query_vector)

    # creates a {word: word_count} for tokens in query
    query_token_count_dict = Counter(processed_query_tokens)

    # number of tokens in query, total word count in query document
    query_word_count = len(processed_query_tokens)
    # print("Query Word Count: ", query_word_count)

    # main process here
    for token in np.unique(processed_query_tokens):
        # calculates term frequencies of all tokens of query doc
        query_tf = query_token_count_dict[token] / float(query_word_count)

        # calculates document_frequency of token
        query_df = 0
        try:
            query_df = DOCUMENT_FREQUENCY[token]
        except:
            pass

        ''' for testing purpose: shows calculated document frequency '''
        # print("query_df: ", query_df)

        # calculates inverse document frequency for all tokens of query doc
        query_idf = math.log((NO_OF_FILES + 1) / float(query_df + 1))

        # creating query vector using query idf values
        try:
            word_index = UNIQUE_WORD_LIST.index(token)
            query_vector[word_index] = query_tf * query_idf
        except:
            pass

    ''' for testing purpose: shows generated query vector '''
    # print("Generated Query Vector: ", query_vector)
    # print("Generated Query Vector (Magnitude): ", np.linalg.norm(query_vector))

    return query_vector


def calculate_cosine_similarity(doc_vec_1, doc_vec_2):
    ''' calculates and returns cosine similarity between two document vectors '''
    cosine_similarity = np.dot(doc_vec_1,doc_vec_2) / (np.linalg.norm(doc_vec_1) * np.linalg.norm(doc_vec_2))
    return cosine_similarity


def get_relevant_docs_cosine_similarity(query, document_vectors, k = MAX_RELEVANT_DOCS):
    ''' returns list of k- most relevant docs for given query '''

    ''' GLOBAL VARIABLES '''
    global UNIQUE_WORD_LIST
    global DOCUMENT_FREQUENCY
    global UNIQUE_WORD_COUNT

    # first generate query vector for given query
    print("   Generating Query Vector ..."),
    query_vector = generate_document_vector(query)
    print("DONE.")
    # print("Query Vector: ")
    # print(query_vector)

    cos_sim = []
    count = 1
    # calculate cosine similarity of query with all other documents
    for document_vector in document_vectors:
        print("   Calculating Cosine Similarity with Document #" + str(count) + "\r"),
        cos_sim.append(calculate_cosine_similarity(query_vector, document_vector))
        count += 1
    # print("Cosine Similarity of Query with all documents: ")
    # print(cos_sim)

    # obtain k- most relevant document index
    out = np.array(cos_sim).argsort()[-k:][::-1] + 1
    # print("\n\nRelevant docs: ")
    # print(out)

    return out


def create_save_document_vectors(word_freq_dict_list_2):
    ''' creates and saves document vectors in CSV format '''

    ''' GLOBAL VARIABLES '''
    global NO_OF_FILES
    global DOC_VEC_FILE
    global DOC_VEC_LENGTH_FILE

    print("Pre-computed document vectors file not found")
    print(" Calculating Document Vectors for all document ..."),
    document_vectors = create_document_vector(word_freq_dict_list_2)
    print("DONE.")

    print(" Saving calculated Document Vectors to CSV ..."),
    data = asarray(document_vectors)
    savetxt(DOC_VEC_FILE, data, delimiter=',')
    print("DONE.\n")

    doc_vec_length_file = open(DOC_VEC_LENGTH_FILE, "w+")
    doc_vec_length_file.write(str(NO_OF_FILES))
    doc_vec_length_file.close()

    return document_vectors


def main():
    ''' MAIN FUNCTION starts here '''
    header()

    ''' GLOBAL VARIABLES '''
    global UNIQUE_WORD_LIST
    global UNIQUE_WORD_COUNT
    global DOCUMENT_FREQUENCY
    global NO_OF_FILES
    global NO_OF_QUERY_DOCS
    global TF_IDF_LIST
    global PRECISION_RECALL_DIR
    global RELEVANT_DOCS_LIST_FILE

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
            tfidf_file.close()
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
    print("\n\nFinding Relevant Documents ...\n------------------------------")
    print(" Reading Query Documents ..."),
    query_docfile = open("query_documents", 'r')
    query_docs = [None] * NO_OF_QUERY_DOCS

    ''' reading query documents '''
    for index in range(NO_OF_QUERY_DOCS):
        query_docs[index] = query_docfile.readline()

    query_docfile.close()
    print("DONE.")

    ''' for all QUERY DOCS '''
    relevant_docs_res = [None] * NO_OF_QUERY_DOCS

    ''' Finding relevant documents for queries '''
    print("\n Finding relevant documents for queries ...")

    for index in range(NO_OF_QUERY_DOCS):
        print("  Matching relevant documents for Query Doc " + str(index+1) + " ..."),
        relevant_docs_res[index] = get_matching_docs(tf_idf_list, query_docs[index])
        print("DONE.")

    print(" DONE.")

    print("\n\nRESULT: RELEVANT DOCUMENT FOR QUERIES")
    print("-------------------------------------")
    for index in range(NO_OF_QUERY_DOCS):
        print("\nMatched relevant documents for Query Doc  " + str(index+1))
        print(relevant_docs_res[index])


    ''' Precision and Recall Part '''
    print("\n\nPRECISION AND RECALL PART\n-------------------------")
    relevant_docs_all = [None] * NO_OF_QUERY_DOCS

    if os.path.exists(RELEVANT_DOCS_LIST_FILE):
        ''' get relevant doc ids from file '''
        relevant_docs_list_file = open(RELEVANT_DOCS_LIST_FILE, "r")
        for index in range(NO_OF_QUERY_DOCS):
            relevant_docs_all[index] = eval(relevant_docs_list_file.readline())
        relevant_docs_list_file.close()

        ''' for testing purpose only '''
        # print(relevant_docs_all)
        # print(type(relevant_docs_all[0]))

        ''' for top k- most ranked documents '''
        precision_recall_cases = [10, 50, 100, 500]

        ''' displays precision and recall score for each query, does not save score '''
        for case in precision_recall_cases:
            precision_recall_filename = PRECISION_RECALL_DIR + "case_top_" + str(case) + ".result"
            precision_recall_file = open(precision_recall_filename, "w+")

            # number of document to retrieve
            num_retrieved = int(case)

            sum_precision = 0.0
            sum_recall = 0.0

            print("\nFOR CASE " + str(case) + "\n-----------")
            precision_recall_file.write("\n" + "For Case " + str(case))

            for index in range(NO_OF_QUERY_DOCS):
                # number of document to retrieve
                num_relevant = len(relevant_docs_all[index])

                # get matching document
                relevant_docs_res[index] = get_matching_docs(tf_idf_list, query_docs[index], case)

                # find relevant document among retrieved documents
                relevant_retrieved_list = [x for x in relevant_docs_res[index] if x in relevant_docs_all[index]]
                num_relevant_retrieved = len(relevant_retrieved_list)

                # display result
                print("\n For Query #" + str(index+1))
                print("  # of Retrieved Docs: " + str(num_retrieved))
                print("  # of Relevant Docs: " + str(num_relevant))
                print("\n  Relevant Docs: " + str(relevant_docs_all[index]))
                print("  Relevant Docs Retrieved: " + str(relevant_retrieved_list))
                print("  # of Relevant Docs Retrieved: " + str(num_relevant_retrieved))

                # calculate precision and recall
                precision = num_relevant_retrieved / float(num_retrieved)
                recall = num_relevant_retrieved / float(num_relevant)

                # display precision and recall
                print("\n  Precision for Query " + str(index+1) + ": " + str(precision))
                print("  Recall for Query " + str(index+1) + ": " + str(recall))
                print("\t\t\t* * *")
                sum_precision += precision
                sum_recall += recall


                ''' writing to the file '''
                precision_recall_file.write("\n For Query #" + str(index+1))
                precision_recall_file.write("\n" + "  # of Retrieved Docs: " + str(num_retrieved))
                precision_recall_file.write("\n" + "  # of Relevant Docs: " + str(num_relevant))
                precision_recall_file.write("\n" + "\n  Relevant Docs: " + str(relevant_docs_all[index]))
                precision_recall_file.write("\n" + "  Relevant Docs Retrieved: " + str(relevant_retrieved_list))
                precision_recall_file.write("\n" + "  # of Relevant Docs Retrieved: " + str(num_relevant_retrieved))
                precision_recall_file.write("\n" + "\n  Precision for Query " + str(index+1) + ": " + str(precision))
                precision_recall_file.write("\n" + "  Recall for Query " + str(index+1) + ": " + str(recall))
                precision_recall_file.write("\n\t\t\t* * *\n")

            # calculate and display average precision and recall
            avg_precision = sum_precision / NO_OF_QUERY_DOCS
            avg_recall = sum_recall / NO_OF_QUERY_DOCS

            print("\n AVG Precision: " + str(precision))
            print(" AVG Recall: " + str(precision))
            print(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")

            ''' writing to the file '''
            precision_recall_file.write("\n" + " AVG Precision: " + str(precision))
            precision_recall_file.write("\n" + " AVG Recall: " + str(precision))
            precision_recall_file.write("\n * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")

            precision_recall_file.close()
    else:
        print(" Relevant Docs File not found.")

    print("\nPRECISION AND RECALL PART END.")

    ''' MATCHING RELEVANT DOCUMENT USING COSINE SIMILARITY '''
    print("\n\nMatching relevant documents using Cosine Similarity ...")
    print("-------------------------------------------------------")
    UNIQUE_WORD_LIST = word_dict_all_2.keys()
    UNIQUE_WORD_COUNT = len(UNIQUE_WORD_LIST)
    TF_IDF_LIST = copy.deepcopy(tf_idf_list)
    # print("UNIQUE_WORD_COUNT:",UNIQUE_WORD_COUNT)

    ''' added on 15 JAN, save document vectors in CSV '''
    if not os.path.exists(DOC_VEC_FILE):
        document_vectors = create_save_document_vectors(word_freq_dict_list_2)
    else:
        if not os.path.exists(DOC_VEC_LENGTH_FILE):
            document_vectors = create_save_document_vectors(word_freq_dict_list_2)
        else:
            doc_vec_length_file = open(DOC_VEC_LENGTH_FILE, "r")
            doc_vec_row_count = doc_vec_length_file.readline()
            doc_vec_length_file.close()
            doc_vec_row_count = int(doc_vec_row_count) if doc_vec_row_count else 0

            if doc_vec_row_count < NO_OF_FILES:
                document_vectors = create_save_document_vectors(word_freq_dict_list_2)
            else:
                print("Pre-computed document vectors file found.\n Reading CSV ..."),
                document_vectors = loadtxt(DOC_VEC_FILE, delimiter=',')
                print("DONE.\n")

    relevant_docs_cossim_list = [None] * NO_OF_QUERY_DOCS

    print(" Calculating Document Frequency for all documents ..."),
    DOCUMENT_FREQUENCY = computeDF(word_freq_dict_list_2, word_dict_all_2)
    print("DONE.")

    print("\n Finding relevant documents for queries ...")
    for index in range(NO_OF_QUERY_DOCS):
        print("  Finding relevant documents for query #" + str(index + 1) + " ...")
        query_doc = query_docs[index]
        relevant_docs_cossim_list[index] = get_relevant_docs_cosine_similarity(query_doc, document_vectors)
        print("\n  DONE.\n")
    print("DONE.")

    print("\n\nRESULT: RELEVANT DOCUMENT FOR QUERIES (USING COSINE SIMILARITY)")
    print("---------------------------------------------------------------")
    for index in range(NO_OF_QUERY_DOCS):
        print("\nMatched relevant documents for Query #" + str(index + 1))
        # print(" " + str(query_docs[index].split("\n")[0]))
        print(list(relevant_docs_cossim_list[index]))


    ''' Precision and Recall Part (Cosine Similarity) here '''
    print("\n\nPRECISION AND RECALL PART (COSINE SIMILARITY)\n---------------------------------------------")
    relevant_docs_all = [None] * NO_OF_QUERY_DOCS

    if os.path.exists(RELEVANT_DOCS_LIST_FILE):
        ''' get relevant doc ids from file '''
        relevant_docs_list_file = open(RELEVANT_DOCS_LIST_FILE, "r")
        for index in range(NO_OF_QUERY_DOCS):
            relevant_docs_all[index] = eval(relevant_docs_list_file.readline())
        relevant_docs_list_file.close()

        ''' for testing purpose only '''
        # print(relevant_docs_all)
        # print(type(relevant_docs_all[0]))

        ''' for top k- most ranked documents '''
        precision_recall_cases = [10, 50, 100, 500]

        sum_precision = 0.0
        sum_recall = 0.0

        relevant_docs_cossim_list = [None] * NO_OF_QUERY_DOCS

        ''' displays precision and recall score for each query, does not save score '''
        for case in precision_recall_cases:
            precision_recall_filename = PRECISION_RECALL_DIR + "_cossim_case_top_" + str(case) + ".result"
            precision_recall_file = open(precision_recall_filename, "w+")

            ''' for testing purpose only '''
            # print(relevant_docs_cossim_list)
            # relevant_docs_res[index] = get_matching_docs(tf_idf_list, query_docs[index], case)

            sum_precision = 0.0
            sum_recall = 0.0

            precision_recall_file.write("COSINE SIMILARITY | PRECISION AND RECALL\n")

            print("FOR CASE " + str(case) + "\n------------")
            precision_recall_file.write("\n" + "For Case " + str(case))

            for index in range(NO_OF_QUERY_DOCS):
                # number of document to retrieve
                num_relevant = len(relevant_docs_all[index])

                print("\n For Query #" + str(index+1))

                # find matching document using cosine similarity
                print("  Finding " + str(case) + "- most relevant documents for query #" + str(index + 1) + " ...")
                relevant_docs_cossim_list[index] = get_relevant_docs_cosine_similarity(query_docs[index], document_vectors, case)

                ''' for testing purpose only '''
                # print("")
                # print(relevant_docs_cossim_list[index])
                # print(list(relevant_docs_cossim_list[index]))

                print("\n  DONE.\n")

                # find relevant document among retrieved documents
                relevant_retrieved_list = [x for x in list(relevant_docs_cossim_list[index]) if x in relevant_docs_all[index]]
                num_relevant_retrieved = len(relevant_retrieved_list)

                # display result
                print("  # of Retrieved Docs: " + str(num_retrieved))
                print("  # of Relevant Docs: " + str(num_relevant))
                print("\n  Relevant Docs: " + str(relevant_docs_all[index]))
                print("  Relevant Docs Retrieved: " + str(relevant_retrieved_list))
                print("  # of Relevant Docs Retrieved: " + str(num_relevant_retrieved))

                # calculate precision and recall
                precision = num_relevant_retrieved / float(num_retrieved)
                recall = num_relevant_retrieved / float(num_relevant)

                # display precision and recall
                print("\n  Precprecision and recallision for Query " + str(index+1) + ": " + str(precision))
                print("  Recall for Query " + str(index+1) + ": " + str(recall))
                print("\t\t\t* * *")
                sum_precision += precision
                sum_recall += recall


                ''' writing to the file '''
                precision_recall_file.write("\n For Query #" + str(index+1))
                precision_recall_file.write("\n" + "  # of Retrieved Docs: " + str(num_retrieved))
                precision_recall_file.write("\n" + "  # of Relevant Docs: " + str(num_relevant))
                precision_recall_file.write("\n" + "\n  Relevant Docs: " + str(relevant_docs_all[index]))
                precision_recall_file.write("\n" + "  Relevant Docs Retrieved: " + str(relevant_retrieved_list))
                precision_recall_file.write("\n" + "  # of Relevant Docs Retrieved: " + str(num_relevant_retrieved))
                precision_recall_file.write("\n" + "\n  Precision for Query " + str(index+1) + ": " + str(precision))
                precision_recall_file.write("\n" + "  Recall for Query " + str(index+1) + ": " + str(recall))
                precision_recall_file.write("\n\t\t\t* * *\n")

            # calculate and display average precision and recall
            avg_precision = sum_precision / NO_OF_QUERY_DOCS
            avg_recall = sum_recall / NO_OF_QUERY_DOCS

            print("\n AVG Precision: " + str(precision))
            print(" AVG Recall: " + str(precision))
            print(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")

            ''' writing to the file '''
            precision_recall_file.write("\n" + " AVG Precision: " + str(precision))
            precision_recall_file.write("\n" + " AVG Recall: " + str(precision))
            precision_recall_file.write("\n * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n")

            precision_recall_file.close()
    else:
        print(" Relevant Docs File not found.")

    print("\nPRECISION AND RECALL PART (COSINE SIMILARITY) END.")

    print("\nDONE.")


if __name__ == "__main__":
     main()
     # raw_input()
