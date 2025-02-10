import mysql.connector
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import MYSQL_CONFIG
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class RAGMySQL:
    
    client = OpenAI()
    
    def __init__(self):
        self.db = mysql.connector.connect(**MYSQL_CONFIG)
        self.cursor = self.db.cursor(dictionary=True)
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

    def retrieve_data(self, query):
        """Fetch relevant data from MySQL based on a user query. The table names are 'users' and 'products'."""
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            return f"Database Error: {err.msg}"
    
    def generate_response(self, question, context):
        """Use OpenAI to generate an answer using retrieved data."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Context: {context}\n\nUser Question: {question}\n\nAnswer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(context=context, question=question)

    def query_rag(self, question):
        """Convert user question into an SQL query, fetch data, and generate a response."""
        
        # Define a better prompt to ensure valid SQL queries
        sql_prompt = f"""
        Convert the following natural language question into a MySQL query.
        Use only these tables: 'users' and 'products'.
        - 'users' table columns: id, username, email, created_at.
        - 'products' table columns: id, username, Category, Description, Price, Stock, created_at, username (FK to users).
        Ensure proper SQL syntax and use INNER JOIN where needed.

        Natural Language Question: {question}
        SQL Query:
        """

        # Generate SQL query using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": sql_prompt}]
        )
        sql_query = response.choices[0].message.content.strip()

        # Validate the generated SQL query before execution
        allowed_keywords = ["SELECT", "FROM", "JOIN", "WHERE", "LIMIT", "ORDER BY"]
        if not any(keyword in sql_query.upper() for keyword in allowed_keywords):
            return "Error: Generated SQL query seems incorrect. Please rephrase your question."

        try:
            print(f"Generated SQL Query: {sql_query}")  # Debugging
            retrieved_data = self.retrieve_data(sql_query)
            if retrieved_data:
                context = str(retrieved_data)
                return self.generate_response(question, context)
            else:
                return "No relevant data found in the database."
        except Exception as e:
            return f"Unexpected error: {str(e)}"
