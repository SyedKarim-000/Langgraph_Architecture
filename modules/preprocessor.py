from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI  # or any other LLM you're using

# Prompt to extract keywords



# Preprocessor class
class Preprocessor:
        

    def extract_keywords(self, query: str) -> list[str]:
        keyword_prompt = PromptTemplate.from_template("""
         You are a keyword extraction assistant.

        Extract only the key entities or phrases from the following query. Return them as a list, one per line. Do not rewrite the query or include explanations.

        Query:
        {query}

        Keywords:
        """)
    #     # LLM chain
        self.keyword_chain = LLMChain(llm=OpenAI(), prompt=keyword_prompt)
        response = self.keyword_chain.run(query=query)
        keywords = [kw.strip().strip('"') for kw in response.split("\n") if kw.strip()]
        return keywords