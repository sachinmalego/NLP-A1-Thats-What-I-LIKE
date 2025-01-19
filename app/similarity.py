import numpy as np

def get_most_similar(word, embeddings, n):
    # Retrieve all words in our embeddings vocabulary
    vocabs = list(embeddings.keys())
    names = ['word', 'sim']

    # Get the embedding vector for the input word, or <UNK> if not found
    vector = embeddings.get(word.lower(), embeddings.get('<UNK>'))
    if vector is None:
        raise ValueError("Word not found and '<UNK>' is not in the embeddings.")

    # Calculate dot products for each word
    similarities = {vocab: np.dot(vector, embeddings[vocab]) for vocab in vocabs}

    # Sort by dot product and retrieve the top `n` words (excluding the input word itself)
    top_n = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_n = [item for item in top_n if item[0] != word][:n]

    # Format the results as a list of dictionaries
    return [dict(zip(names, (vocab, f"{sim:.4f}"))) for vocab, sim in top_n]