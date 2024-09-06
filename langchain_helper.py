from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
import re

load_dotenv()

vectordb_file_path = "faiss_index"

@lru_cache(maxsize=None)
def get_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )

@lru_cache(maxsize=None)
def get_embeddings():
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def create_vector_db():
    loaders = [
        CSVLoader('./knowledge_bank/customer_queries_responses.csv'),
        CSVLoader('./knowledge_bank/ecommerce_products.csv'),
        CSVLoader('./knowledge_bank/policies.csv')
    ]
    
    all_data = []
    for loader in loaders:
        all_data.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_data)

    vectordb = FAISS.from_documents(documents=all_splits, embedding=get_embeddings())
    vectordb.save_local(vectordb_file_path)

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

@lru_cache(maxsize=None)
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, get_embeddings(), allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    system_template = """You are a friendly, helpful, and knowledgeable customer support agent for our e-commerce company, The Hershel's. 
    Your goal is to provide accurate and helpful information to customers based on the context provided. 
    Always maintain a positive and supportive tone in your responses. 
    If you're asked about specific information like email addresses, phone numbers, or addresses, make sure to include that information in your response if it's available in the context.
    If you're unsure about something, it's okay to admit that and offer to connect the customer with a human agent for further assistance."""

    human_template = """Given the following context and a question, generate an answer based on this context only.
    If the question is related to a product, provide the information about the product. If you don't find the product, politely inform the customer that the product is not in our current inventory.
    If the question is related to the company, policies, contact info, or payment info, look for the answer in the "response" if the question is related to the "query". If you don't find any information, apologize and explain that you couldn't find the specific information in your knowledge base.
    In the answer, please try to provide as much relevant information as possible from the "response" section in the source document context.
    If you don't find a direct answer in the context, use your understanding to rephrase the question and look for related information. Then provide a helpful response based on the available information.
    If the answer is not found in the context and you can't provide a helpful response, kindly state "I apologize, but I don't have enough information to answer that question accurately. Would you like me to connect you with a human customer support representative for more assistance?".
    Do not make up something. Always answer based on your knowledgebase.

    CONTEXT: {context}

    QUESTION: {question}

    Please provide a friendly and helpful response:"""

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

def answer_question(question):
    chain = get_qa_chain()
    response = chain(question)
    answer = response['result']
    
    # Check if the answer contains an email address, if not, try to extract it from the source documents
    if not extract_email(answer):
        for doc in response['source_documents']:
            email = extract_email(doc.page_content)
            if email:
                answer += f"\n\nFor further assistance, you can contact us at: {email}"
                break
    
    return answer

if __name__ == "__main__":
    create_vector_db()