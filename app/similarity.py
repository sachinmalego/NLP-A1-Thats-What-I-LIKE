import numpy as np

# function to find the most similar word to the input vector
def get_most_similar(word, embeddings, n):
    # retrieve all words in our embeddings vocabs
    vocabs = list(embeddings.keys())
    names = ['word', 'sim']

    try:
        vector = embeddings[word.lower()]
    except:
        vector = embeddings['<UNK>']
    
    similarities = {}

    # for each word in the vocabs, find the cosine similarities between word vectors in our embeddings and the input vector
    for vocab in vocabs:
        sim = np.dot(vector, embeddings[vocab])
        similarities[vocab] = "{:.4f}".format(sim)

    top_n = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[1:n+1]

    return [dict(zip(names, val)) for val in top_n]