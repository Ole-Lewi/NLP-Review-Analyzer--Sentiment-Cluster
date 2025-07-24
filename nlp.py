import pandas as pd

df = pd.read_csv('1429_1.csv')

print(df["reviews.text"][0])

#drop missing reviews
df = df.dropna(subset=['reviews.text'])

#drop duplicates
df = df.drop_duplicates(subset=['reviews.text'])

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Tokenize the reviews
tokens = df['reviews.text'].apply(word_tokenize)

# Convert tokens to lowercase
tokens = [[token.lower() for token in review_tokens] for review_tokens in tokens]

#Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Remove stopwords
tokens = [[token for token in review_tokens if token not in stop_words] for review_tokens in tokens]

#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
tokens = [[lemmatizer.lemmatize(token) for token in review_tokens] for review_tokens in tokens]

#Join tokens back to text
df['cleaned_reviews'] = [' '.join(review) for review in tokens]

#Vectorization with TF-IDF(Term Frequency-Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the cleaned reviews
X = vectorizer.fit_transform(df['cleaned_reviews'])

#Clustering with KMeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5, random_state=42)
# Fit the model to the TF-IDF matrix
clusters = model.fit_predict(X)

# Add the cluster labels to the DataFrame
df['cluster'] = clusters

#Peek into review per cluster - helps understand the topics in each cluster
for i in range(model.n_clusters):
    print(f"Cluster {i} Reviews:\n")
    print(df[df['cluster'] == i]['cleaned_reviews'].head(5).to_string(index=False))
    print("\n")

#Get the top terms per cluster
import numpy as np

# Get the top words per cluster
def get_top_keywords(X, labels, vectorizer, n_terms=10):
    df = pd.DataFrame(X.todense()).groupby(labels).mean()
    terms = vectorizer.get_feature_names_out()
    
    for i, row in df.iterrows():
        print(f"\nCluster {i} top words:")
        print(', '.join([terms[t] for t in np.argsort(row)[-n_terms:]]))

get_top_keywords(X, clusters, vectorizer)

#Visualization of clusters using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='rainbow')
plt.title("Product Review Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
