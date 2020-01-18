# Web Systems and Algorithms (_CSc 559_)

### Pre-requisite
1. Python 2.7.15 ([``https://www.python.org/downloads/release/python-2715/``](https://www.python.org/downloads/release/python-2715/))
2. pip ([``https://pip.pypa.io/en/stable/installing/``](https://pip.pypa.io/en/stable/installing/))

### Installing dependencies
1. Go to ``WSA_Labs`` folder
2. Execute command ``$ pip install -r requirements.txt``

### Running Lab 1
1. Go to ``WSA_Labs`` folder (if not in ``WSA_Labs`` folder)
2. Run ``$ python wsa.py``

#### _Notes:_
1. ``cranfieldDocs`` directory contains 1400 documents used in this program.

2. ``tfs`` directory contains calculated _TF values_ separate for each document.

3. ``tfidfs`` directory contains calculated _TF-IDF values_ separate for each document.

4. ``precision_recall_result`` directory contains output of _precision and recall_ calculation in separate file for each case.

5. ``query_documents`` file contains one query per line

6. ``relevant_doc_ids`` file contains document ids of relevant document per line for each query in `` query_documents`` file

7. ``requirements.txt`` lists dependencies required for ``wsa.py``
