from config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, DB_GENERAL_DATA, TABLE_GENERAL_DATA, SCHEMA_GENERAL_DATA,\
    DB_GENE_ANNOT, TABLE_GENE_ANNOT, SCHEMA_GENE_ANNOT, VECTOR_DB_DIR, CHAT_MODEL,\
    SYSTEM_MESSAGE_SQL, SYSTEM_MESSAGE_EHR , EMBEDDING_DIM

import chatbot as cb
import os


# Directory for storing the vector database and metadata
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

#data_dir = "/Users/dajanapanovic/PycharmProjects/pythonProject/HR_data"
#create_vector_database(data_dir, client)

# Run the chatbot
if __name__ == "__main__":
    cb.chatbot()


# /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install -qU langchain-openai
