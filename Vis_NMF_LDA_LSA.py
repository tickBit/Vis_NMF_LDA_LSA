"""
Little program to compare the results of NMF, LDA and LSA topic models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import numpy as np
import spacy
import requests

# can be changes to "en_core_web_sm", if needed
nlp = spacy.load("en_core_web_lg")

# SPARQL query to get the abstract of Isaac Newton from DBPedia
endpoint_url = "http://dbpedia.org/sparql"
query = '''
    SELECT *
    WHERE {
            ?scientist  rdfs:label      "Isaac Newton"@en ;
            dbo:abstract  ?abstarct .
            FILTER ( LANG ( ?abstarct ) = 'en' )
        }
'''

r = requests.get(endpoint_url, params = {'format': 'json', 'query': query})
data = r.json()

text = data["results"]["bindings"][0]["abstarct"]["value"]
doc = nlp(text)

# Create a list of sentences from the abstract
document = []
for sent in doc.sents:
    document.append(sent.text)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
A = vectorizer.fit_transform(document).toarray()

# Create a TruncatedSVD model with 4 components
svd = TruncatedSVD(n_components=6, n_iter=7, random_state=42).fit(A)

# Create a NMF model with 4 components
nmf = NMF(n_components=6, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd').fit(A)

# Create a LDA model with 4 components
lda = LatentDirichletAllocation(n_components=6, max_iter=7, learning_method='online', learning_offset=50.,random_state=42).fit(A)

# Create a list of the topics
topic_names = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6"]

# Create a list of the models
models = [nmf, lda, svd]

# Create a list of the model names
model_names = ["NMF", "LDA", "LSA"]

# Create a list of the model results
model_results = [nmf.transform(A), lda.transform(A), svd.transform(A)]

array_for_sentences = []
# print results
for i in range(len(model_names)):
    #print(model_names[i])
    array_for_sentences.append(model_results[i].argmax(axis=1))

# print the topics from the sentences for each model
for i in range(len(array_for_sentences)):
    print("----------------------------")
    print("--- "+ model_names[i])
    print("----------------------------")
    # repeat until all sentences are printed for all models
    for k in range(len(array_for_sentences[i])):
        printed_topic = -1
        for j in range(len(array_for_sentences[i])):

            if array_for_sentences[i][j] > printed_topic and array_for_sentences[i][j] == k:
                printed_topic = array_for_sentences[i][j]
                print("--- Topic:", (printed_topic+1), "---")

            if array_for_sentences[i][j] == printed_topic:  
                print(document[j])




fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
# x-coordinates from 1 to 6
x = np.arange(1, 7)

fig.suptitle('Topic Models NMF, LDA and LSA for Isaac Newton')
# plot the topic words of NMF
for i, comp in enumerate(nmf.components_):
    words = " ".join([vectorizer.get_feature_names()[j] for j in comp.argsort()[-4:]])
    # plot the topics words as bar chart
    axs[0].bar(x[i], comp.argsort()[-4:], label=words)
    # plot the max 4 topic words in the bar chart
    axs[0].text(x[i], 1, words, horizontalalignment='center', verticalalignment="bottom", rotation=90)
axs[0].set_title("NMF")

# plot the topic words of LDA
for i, comp in enumerate(lda.components_):
    words = " ".join([vectorizer.get_feature_names()[j] for j in comp.argsort()[-4:]])
    # plot the topics words as bar chart
    axs[1].bar(x[i], comp.argsort()[-4:])
    # plot the max 4 topic words in the bar chart
    axs[1].text(x[i], 1, words, horizontalalignment='center', verticalalignment="bottom", rotation=90)
axs[1].set_title("LDA")

# plot the topic words of LSA
for i, comp in enumerate(svd.components_):
    words = " ".join([vectorizer.get_feature_names()[j] for j in comp.argsort()[-4:]])
    # plot the topics words as bar chart
    axs[2].bar(x[i], comp.argsort()[-4:])
    # plot the max 4 topic words in the bar chart
    axs[2].text(x[i], 1, words, horizontalalignment='center', verticalalignment="bottom", rotation=90)
axs[2].set_title("LSA")

plt.show()
