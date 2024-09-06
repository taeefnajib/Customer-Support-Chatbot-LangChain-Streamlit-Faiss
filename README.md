# Customer Support Chatbot with Langchain, Streamlit, and Faiss

This project is a customer support chatbot that uses Langchain for retrieval-augmented generation (RAG) and Faiss for vector database indexing. The chatbot is designed to answer queries based on data stored in CSV files inside the `knowledge_bank` folder, and its frontend is built using Streamlit.

## Features

- **Langchain** for RAG to retrieve relevant responses.
- **Faiss** for efficient similarity search on vector embeddings.
- **Streamlit** as the frontend for easy interaction with the chatbot.
- Queries the data stored in the `knowledge_bank` folder (CSV files).

## Installation

Follow the steps below to set up and run the chatbot locally.

### Step 1: Clone the Repository
```bash
git clone https://github.com/taeefnajib/Customer-Support-Chatbot-LangChain-Streamlit-Faiss
cd Customer-Support-Chatbot-LangChain-Streamlit-Faiss
```
### Step 2: Create a Virtual Environment
Create a virtual environment in the root folder:

```bash
virtualenv venv
```
Activate the virtual environment:
```bash
source venv/bin/activate
```
### Step 3: Install Dependencies
Install the required Python dependencies:

```bash
pip install -r requirements.txt
```
### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory and add your `Google API key`. You can use the `env_template.txt` file as a template to structure your `.env` file.

### Step 5: Run the Application
Start the chatbot application using Streamlit:
```bash
streamlit run main.py
```
## How to Use
Once the application is running, you can input a message into the chatbot.
The chatbot will respond based on the data it has learned from the CSV files located in the `knowledge_bank` folder.

`Faiss` is used as the vector database to quickly find relevant data points.

`Langchain` powers the retrieval-augmented generation (RAG) system, ensuring accurate and context-aware responses.

### Technologies Used
* Langchain: For RAG-based responses.
* Faiss: For vector similarity search.
* Streamlit: For building the user interface and interacting with the chatbot.
* Google API: For additional functionality (set up via the .env file).

### Folder Structure
```plaintext
├── faiss_index/
│   ├── index.faiss        
│   ├── index.pkl       
├── knowledge_bank/  
│   ├── customer_queries_responses.csv     
│   ├── ecommerce_products.csv 
│   ├── policies.csv  
├── .gitignore  
├── env_template.txt 
├── langchain_helper.py 
├── LICENSE            
├── main.py   
├── README.md        
├── requirements.txt
```
## License
This project is licensed under the MIT License. See the LICENSE file for more details.