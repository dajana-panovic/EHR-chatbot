from config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, DB_GENERAL_DATA, TABLE_GENERAL_DATA, SCHEMA_GENERAL_DATA,\
    DB_GENE_ANNOT, TABLE_GENE_ANNOT, SCHEMA_GENE_ANNOT, VECTOR_DB_DIR, CHAT_MODEL,\
    SYSTEM_MESSAGE_SQL, SYSTEM_MESSAGE_EHR , EMBEDDING_DIM
import vector_database as vdb

from openai import OpenAI
import sqlite3 as sql
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

client = OpenAI()

def generate_question_embedding(client: OpenAI, md_question: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=md_question
    )

    embedding = np.array(response.data[0].embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm != 0:
        normalized_embedding = embedding / norm
    else:
        normalized_embedding = 0
        print("Embedding norm = 0. Can not proceed.")

    return normalized_embedding

def generate_answer_from_EHR(client, md_question: str, EHR: str) -> str:

    system_message = (
        "You are a helpfull assistent."
        f"Here is the full EHR for a given patient: {EHR}"
        "Could you please answer the question for this patient?"
        "Please be aware that there will be provided the whole conversation history. "
        "Only the last question should be answered."
    )

    # Generate answer
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": md_question}
        ],
        temperature=0
    )
    answer = completion.choices[0].message.content

    return answer


def ehr_retrieval(client, md_question: str) -> str:
    # Load vector data
    index, metadata = vdb.load_vector_database()
    query_embedding = generate_question_embedding(client, md_question)

    # Retrieve the top-k most relevant results. I will only take the first most relevant.
    # This approach is prone to error. It's better to filter using metadata.
    k = 1
    distances, indices = vdb.query_vector_database(query_embedding, index, k)

    EHR = vdb.extract_the_EHR_content(indices, distances, metadata)
    question_answer = generate_answer_from_EHR(client, md_question, EHR)

    return question_answer

def process_question(client: OpenAI, question: str) -> str:
    # based on the nature of the question the function should return which function should be called
    # there is option A: ehr_retrieval and option B: execute_sql_query

    system_message = (
        f"You are a helpful assistant. Based on the given question "
        "decide which function we should use in order to answer the question."
        "There will be provided the full conversation history. The question that should be answered is the last one."
        "There are only 2 functions: ehr_retrieval and execute_sql_query."
        "ehr_retrieval answers questions using electronic health records of a given patient. All question related to diagnosys,"
        "previous doctors visits, family history, therapies and similar should be answered using this function."
        "execute_sql_query is a function that is used when we want to get answers on general patient data which is"
        f"data from {SCHEMA_GENERAL_DATA} OR data related to genes and variants from {SCHEMA_GENE_ANNOT}."
        f"Your answer should be only a string - function name. So it's either execute_sql_query or ehr_retrieval"
    )

    # Let LMM decide which function should be used

    model = ChatOpenAI(model=CHAT_MODEL)
    prompt_template = ChatPromptTemplate([
         ("system", system_message),
         ("user", "{user_message}")
     ])

    chain = prompt_template | model | StrOutputParser()

    function_to_be_called = chain.invoke({"user_message": question}).strip()

    return function_to_be_called

def sql_query(client, user_query, data_schema, table_name):

    system_message = (
        "You are an expert in database querying using sqlite."
        "Please generate SQL query that will help user query it's database."
        "The user will specify what he wants to know. There will be included the whole conversation history."
        "Please beaware of the context but focus on the last question. This one is of our interest."
        "He working with this data:"
        f"Schema: {data_schema}, Table: {table_name}."
        "The response should return only sql query without any additional explanations "
        "and it should be in a format that is suitable to directly be an input to execute() function"
        "from sqlite package. for example SELECT Trait FROM variant_annotations WHERE Gene = 'BRCA1'"

    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
        ],
        temperature=0
    )
    return str(response.choices[0].message.content).strip().encode('utf-8').decode('utf-8')



def execute_sql_query(client, database_name, natural_language_query, data_schema, table_name):

    connection = sql.connect(database_name)
    cursor = connection.cursor()
    cursor.execute(sql_query(client, natural_language_query, data_schema, table_name).strip())
    result = cursor.fetchall()
    connection.close()


    # result return is not that human readable. Want to fix that so this is why system message is like this. I have to improve this.
    system_message = (
        f"""You are an expert in data wrangling and data extraction."
        "I will give you a string which is a sentence, or a word."
        "based on the {natural_language_query} conclude what should be"
        "included in the response and format that response so that it doesn't contain"
        "that much brackets. The response should be correct and concise without unnecessary information/descriptions
        on the process how we generated or formatted the response. Also, it should contain only information from {result}."
        """

    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": str(result)},
        ],
        temperature=0
    )

    return response.choices[0].message.content


def answer_question(client, func_to_be_called: str, question: str) -> str:

    if (func_to_be_called == "execute_sql_query"):  # This should be improved using try/except

        system_message = ("Determine which database to query based on the nature of the last question:"
                          "Consider the last question from the conversation history provided."
                          "Based on the last question determine which database to query."
                          "General Patient Data:"
                          "If the last question is about a specific patient (e.g., demographics, medical history, or test results), "
                          "query the general_data database. Please be aware of the conversation history and determin which patient we are "
                          "talking about even if not stated. If multiple patients provided consider the last one."
                          "Variant and Gene Information:"
                          "If the last question concerns variants, genes, or annotations (e.g., gene-specific data or variant interpretations), "
                          "query the annotation database."
                          "Respond to the query with the following rules:"
                          "For questions about specific patient data, return results only from the general_data database."
                          "For questions about variants and genes, return results only from the annotation database."
                          "Return only the requested strings (general_data or annotation), without adding any additional information or explanations.")


        data_table = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message
                                              },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        data_table = data_table.choices[0].message.content

        if data_table == 'general_data':
            try:
                return execute_sql_query(client, DB_GENERAL_DATA, question, SCHEMA_GENERAL_DATA, TABLE_GENERAL_DATA)
            except:
                print('Information not present in general_patient_data table.')
        elif data_table == 'annotation':
            try:
                return execute_sql_query(client, DB_GENE_ANNOT, question, SCHEMA_GENE_ANNOT, TABLE_GENE_ANNOT)
            except:
                print('Information not present in variant_annotation table.')

    elif (func_to_be_called == "ehr_retrieval"):
        try:
            return ehr_retrieval(client, question)
        except:
            print('There is not that information in EHR data.')
    else:
        print("There is an error. Request can't be handled.")


def chatbot():

    text = (
        "Hello. I will assist you in retrieving information about the patients you are interested in. "
        "Please ask me a question.\n"
        "Type 'exit' to quit."
    )
    print(text)

    conversation_history = []  # To store the conversation history

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Chatbot: Conversation end.")
            break

        # Include the conversation history in the process
        context = "\n".join(
            [f"You: {entry[0]}\nChatbot: {entry[1]}" for entry in conversation_history]
        )

        # Combine the history with the current user input for context-aware processing
        full_input = f"{context}\nYou: {user_input}"

        function_to_be_called = process_question(client, full_input)
        response = answer_question(client, function_to_be_called, full_input)

        conversation_history.append((user_input, response))

        print(f"Chatbot: {response}")

