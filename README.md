# Information Retrieval with tf-idf and BM25 ranking algorithm
This project focuses on implementing document retrieval system using tf-idf and BM25 ranking algorithm

## Description:
In this project, the task is to develop and evaluate a two-stage information retrieval model that given a query returns the n most relevant documents and then ranks the sentences within
the documents. In the beginning, a baseline document retriever with tf-idf features is implemented. Afterwards to improve the baseline retrieval system,  the document retriever with an advanced approach like BM25 is used.

- 💡 Goal:  Retrieve relevant document based on given query

- Dataset
  - 📊 Corpus: `TREC` dataset
  - ❓Query: A text file that contains the query
  - 🛠️ Helper file: A text file contains regular expression for each query for evaluation
  - ⚙️ Method: TF- IDF
    - Process: The corpus is parsed using `beautifulsoup` library to create a data frame(`Pandas`) contains document id and corresponding text
    - Preprocessing of the text is done by
      - Tokenization(using `NLTK` library)
      - Lower-casing
      - Removing punctuation tokens
    - `Inverse document frequency` is computed for each term(Using `Math` library to calculate the log value)
    - `Term frequency` is calculated for each term of a document and stored along with `doc_id`
    - For query terms, the tf-idf weights for these terms as the product of the term’s idf and the tf-value of the term in the respective document is computed and stored
    - Each corpus document and each query document is represented as vector of tf-idf score
    - **Cosine similarity** is calculated to get the relevance
    - The relevance score is **sorted and top 50** documents are output as result
  - 📈 Evaluation:
    - Gold standard relevant documents are fetched by using the `regex` in the pattern file
    - **Evaluation metric:**
      - `Precision@50`: **65%**
    - Disadvantage: Does not capture position in text, semantics, co-occurrences in different documents
  - ⚙️ Method: Okapi BM25
    - Process: Calculation of tf and idf weight is similar as before
    - 2 new hyperparameters:
      - K = controls the impact of term frequency(1.2)
      - B = controls the impact of document frequency(0.75)
      - The formula is different.
  - 📈 Evaluation:
    - `Precision @50` improves to **79%**

      
## 🚀 How to run the project:
* Libraries required: BeautifulSoup or `xml`, `NLTK`.
* There are two python files: `tf-idf.py`, `bm25.py`.
* Open the file using your favourite python framkework.
* Keep the `trec_documents`, `patterns` and `test_questions` in the same folder along with the python files.
* Prefereably start with the `tf_idf.py` file as this is the baseline.
* With eventual complexities: `Open BM25.py`  file for improved result.
* Each file contains detailed clear description of the functionalities.

