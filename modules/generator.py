import os
import openai
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Generator:
    llm:ChatOpenAI
    prompt:ChatPromptTemplate
    def __init__(self,prompt):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.prompt = prompt


    def generate(self, context, query):
        # document_chain=create_stuff_documents_chain(self.llm,self.prompt)
        # retrieval_chain=create_retrieval_chain(context,document_chain)
        # response =  retrieval_chain.invoke({"input":query})
        # return response['answer']
        prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}"
        prompt2 = f'''Answer based on the following context: \n{context}\n\nQuestion: {query}
        If the portfolio consists of primarily of bank loans, populate ARBITRAGE CASH FLOW - HY CLO 
        If the portfolio consists primarily of Small & middle market loans, Populate SMALL AND MIDDLE MARKET'''
        response = openai.chat.completions.create(

            model="gpt-3.5-turbo",

            messages=[{"role": "user", "content": prompt}],

            temperature=0.3

        )

        return response.choices[0].message.content


        