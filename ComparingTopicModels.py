"""
Little program to compare the results of NMF, LDA and LSA topic models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import string
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

# lemmatization for tokenized_document
stemmed_document = []
for sent in document:
    temp = []
    for token in nlp(sent):
        temp.append(token.lemma_)
    stemmed_document.append(temp)

# spaCy's stopword list
stopwords = spacy.lang.en.stop_words.STOP_WORDS

tokenized_document = []
for sent in stemmed_document:
    temp = []
    for token in sent:
        token_nlp = nlp(token)
        if token_nlp.text not in string.punctuation and token_nlp.text not in stopwords:
            temp.append(token_nlp.text)
    tokenized_document.append(temp)

array_of_words_by_sentences = []
for index, word in enumerate(tokenized_document):
    array_of_words_by_sentences.append(' '.join(word))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
A = vectorizer.fit_transform(array_of_words_by_sentences).toarray()

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



# indices of sentences for each model
array_for_sentences = []
# print results
for i in range(len(model_names)):
    #print(model_names[i])
    array_for_sentences.append(model_results[i].argmax(axis=1))


topics = {}
# create a dictionary of topics for each model
for i in range(len(model_names)):
    topics[model_names[i]] = {}
    for j in range(len(topic_names)):
        topics[model_names[i]][topic_names[j]] = []

# add sentences to the topics
for i in range(len(model_names)):
    for j in range(len(topic_names)):
        t = ""
        for k in range(len(array_for_sentences[i])):
            if array_for_sentences[i][k] == j:
                t += document[k] + " "
        topics[model_names[i]][topic_names[j]].append(t.strip())


# calculate the similarity between all the topics of each model
similarity_results = {}
for i in range(len(model_names)):
    for j in range(len(model_names)):
        if i != j:
            similarity_results[model_names[i] + " vs. " + model_names[j]] = {}
            for k in range(len(topic_names)):
                similarity_results[model_names[i] + " vs. " + model_names[j]][topic_names[k]] = {}
                for l in range(len(topic_names)):
                    similarity_results[model_names[i] + " vs. " + model_names[j]][topic_names[k]][topic_names[l]] = nlp(topics[model_names[i]][topic_names[k]][0]).similarity(nlp(topics[model_names[j]][topic_names[l]][0]))


alreadeInComparison = []
# plot the similarity results
for i in range(len(model_names)):
    for j in range(len(model_names)):
        if i != j and model_names[j] + " vs. " + model_names[i] not in alreadeInComparison:
            fig, ax = plt.subplots()
            im = ax.imshow(np.array([[similarity_results[model_names[i] + " vs. " + model_names[j]][topic_names[k]][topic_names[l]] for l in range(len(topic_names))] for k in range(len(topic_names))]), cmap="coolwarm")
            # plot color bar
            cbar = ax.figure.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(topic_names)))
            ax.set_yticks(np.arange(len(topic_names)))
            ax.set_xticklabels(topic_names)
            ax.set_yticklabels(topic_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for k in range(len(topic_names)):
                for l in range(len(topic_names)):
                    text = ax.text(l, k, round(similarity_results[model_names[i] + " vs. " + model_names[j]][topic_names[k]][topic_names[l]], 2), ha="center", va="center", color="w")
            ax.set_title(model_names[i] + " vs. " + model_names[j])
            fig.tight_layout()
            alreadeInComparison.append(model_names[i] + " vs. " + model_names[j])
plt.show()
