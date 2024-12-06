import os

DB_GENERAL_DATA = "General patient data"
TABLE_GENERAL_DATA = "general_patient_data"
SCHEMA_GENERAL_DATA = """
Columns:
  - Patient_ID (string): Patient identification. Patients are de-personalized. Those are codes.
  - Sex (string): Values M for males and F for females.
  - Age (integer): Age of patient in years.
  - Weight (integer): Weight of a patient in kg.
  - Height (integer): Height of a patient in cm.
  - Education (string): Level of education. Primary school, high school, and university.
  - Smoking_Status (string): Indicator if the person is smoking or not. Values: smoker or non-smoker.
  - Physical_Activity (string): Indicator of physical activity level. Values: none, low, medium, high.
  - Alcohol_Consumption (string): Indicator of alcohol consumption. Values: low, moderate, high.
"""

DB_GENE_ANNOT = "Variant annotations"
TABLE_GENE_ANNOT = "variant_annotations"
SCHEMA_GENE_ANNOT = """
Columns:
  - Gene (string): Gene name (e.g., "BRCA1", "TP53").
  - Chromosome (string): The chromosome on which the variant is located (e.g., "chr1", "chrX").
  - Position (integer): The genomic position of the variant on the chromosome.
  - Consequence (string): The predicted biological consequence of the variant (e.g., "missense_variant").
  - Trait (string): The trait or phenotype associated with the variant.
"""

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

VECTOR_DB_DIR = "vector_database"  # Directory to store the FAISS index and metadata
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.json")

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_MESSAGE_SQL = (
        "You are an expert in database querying using SQLite. "
        "Please generate a SQL query to help the user query the database. "
        "Focus only on the last question from the conversation history. "
        "The schema and table details are provided. Respond only with a SQL query that can be used directly with SQLite's execute() function."
    )

SYSTEM_MESSAGE_EHR = (
        "You are a helpful assistant. Based on the question provided, determine which function to call. "
        "Use 'ehr_retrieval' for questions about electronic health records (e.g., diagnosis, family history) "
        "and 'execute_sql_query' for database queries related to general patient data or variant annotations."
    )

#DATA_DIR = "/Users/dajanapanovic/PycharmProjects/pythonProject/HR_data"  # Path to patient data files

# Create the vector database directory if it doesn't exist
# if not os.path.exists(VECTOR_DB_DIR):
#     os.makedirs(VECTOR_DB_DIR)
