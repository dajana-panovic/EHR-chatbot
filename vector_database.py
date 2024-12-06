from config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, EMBEDDING_DIM

import os
import json
import faiss
from openai import OpenAI
import numpy as np

def create_vector_database(data_dir: str, client: OpenAI) -> None:

    # Initialize FAISS index. I will use Flat index since the dataset is small. In the future I maybe change to ANN i.e. IndexIVFFlat
    dim = EMBEDDING_DIM
    index = faiss.IndexFlatIP(dim)
    metadata = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(data_dir, file_name)
            patient_id = file_name.split("_")[0]

            with open(file_path, "r") as f:
                lines = f.readlines()
                full_text = "".join(lines).strip()  # Take all data from file.

                embedding_response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    # I am using this model, but there are also other models that are good for this use case.
                    input=full_text  # E.s. some models trained on PubMed data or HuggingFace have healtcare NLP lib.
                )
                embedding = np.array(embedding_response.data[0].embedding,
                                     dtype=np.float32)  # If we want to save memory we can cast to int but won't do it now

                # Normalize embedding for cosine similarity. This could be done better.
                norm = np.linalg.norm(embedding)
                if norm != 0:
                    embedding = embedding / norm

                # Add normalized embedding to FAISS index. Reshaping to covert 1D vector to 2D vec. Technical thing.
                index.add(embedding.reshape(1, -1))

                # Save metadata
                metadata.append({
                    "patient_id": patient_id,
                    "text": full_text
                })

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata
    with open(METADATA_PATH, "w") as meta_file:
        json.dump(metadata, meta_file)

    print("Vector database created successfully.")

def load_vector_database():
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load metadata
    with open(METADATA_PATH, "r") as meta_file:
        metadata = json.load(meta_file)

    return index, metadata

def query_vector_database(query_embedding, index, k):
    norm = np.linalg.norm(query_embedding)
    if norm != 0:
        query_embedding = np.array(query_embedding / norm, dtype=np.float32)

    # Top-k most similar embeddings. In our usecase we can use k=1
    distances, indices = index.search(query_embedding.reshape(1, -1), k)

    return distances, indices


def extract_the_EHR_content(indices, distances, metadata):  # Indices and distances are the result from query_vector_database function

    top_idx = indices[0][0]
    top_distance = distances[0][0]

    # Validate the index
    if top_idx < len(metadata):
        result = {
            "patient_id": metadata[top_idx]["patient_id"],
            "similarity_score": top_distance,
            "text": metadata[top_idx]["text"]
        }
    else:
        raise ValueError(f"Invalid index {top_idx}. The metadata may be corrupted or inconsistent.")

    return result
