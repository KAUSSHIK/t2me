# knowledge_base.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_embeddings(text_path):
    # Load the combined text
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into sentences
    sentences = text.split('. ')
    sentences = [s.strip() for s in sentences if s.strip() != '']

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = model.encode(sentences)

    # Save sentences and embeddings
    np.save('embeddings.npy', embeddings)
    with open('sentences.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")

    return embeddings, sentences

def create_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    faiss.write_index(index, 'faiss_index.idx')

if __name__ == "__main__":
    embeddings, sentences = create_embeddings('combined_text.txt')
    create_faiss_index(embeddings)
