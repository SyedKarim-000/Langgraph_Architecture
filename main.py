import os
from typing import TypedDict
from dotenv import load_dotenv
from modules.preprocessor import Preprocessor
from modules.retriver import QdrantPDFRetriever
from modules.grader import Grader
from modules.generator import Generator
from modules.rewriter import Rewriter
from modules.evaluator import Evaluator
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()
class PortfolioState(TypedDict):
    query: str
    subqueries: list[str]
    retrieved: list
    score: float
    answer: str
    qury_retry_count: int = 0
   

# Load knowledge base
# with open("data/knowledge_base.txt") as f:
#     docs = f.readlines()

# retriever = Retriever(docs)
retriever = QdrantPDFRetriever()
grader = Grader()
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")
generator = Generator(prompt)
rewriter = Rewriter()
evaluator = Evaluator()
preprocess = Preprocessor()
def preprocess_query_node(state):
    query = state["query"]
    # Example: Add context or remove noise
    enriched_query = preprocess.extract_keywords(query)
    state["subqueries"] = (enriched_query)
    return state

def retrieve_node(state):
    query = state["query"]
    if not query:
        print("❗️ No keywords extracted from the query. Ending process.")
        return "endnode"
    print(f"🔍 Retrieving documents for query: {query}")
    # retrieved = []
    # for subquery in query:
    #     retrieved.append(retriever.retrieve(subquery))
    
    # state["retrieved"] = retrieved
    retrieved = retriever.retrieve(query)
    state["retrieved"] = retrieved
    return state

def grade_node(state):
    score = grader.grade(state["retrieved"])
    state["score"] = score
    return state

def generate_node(state):
    answer = generator.generate(state["retrieved"], state["query"])
    state["answer"] = answer
    print(answer)
    return state

def evaluate_node(state):
    if evaluator.is_answered(state["answer"]):
        return state
    else:
        return state

def rewrite_node(state):
    new_query = rewriter.rewrite(state["query"],state["answer"])
    state["qury_retry_count"] += 1
    state["query"] = new_query
    return state

def conditional_logic(state):
    #check Query retry count
    if state["qury_retry_count"] >= 3:
        print("❗️ Maximum query retries reached. Ending process.")
        return "endnode"
    # Check if the answer is satisfactory
    isAnswerd = evaluator.is_answered(state["answer"])
    if isAnswerd:
        return "endnode"
    else:
        return "rewrite"
def end_node(state):
    print("--------------------END------------------------")

# Build LangGraph
builder = StateGraph(PortfolioState)
builder.add_node("preprocess", preprocess_query_node)
builder.set_entry_point("preprocess")
builder.add_edge("preprocess", "retrieve")

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("generate", generate_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("endnode", end_node)


#builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")
builder.add_edge("grade", "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges("evaluate",conditional_logic, 
                              {"endnode": "endnode", "rewrite": "rewrite"}
                            )
#builder.add_conditional_edges("evaluate",conditional_logic)
builder.add_edge("rewrite", "retrieve")
#builder.add_edge("endnode", "end")
builder.set_finish_point("endnode")

# builder.set_entry_point("retrieve")
# builder.add_edge("retrieve", "generate")
# builder.add_edge("generate", END)
graph = builder.compile()
initial_state = {
    "query": "What is closing date, look for 'closing date/issue date' language ?",
    "subqueries": [],
    "retrieved": [],
    "score": 0.0,
    "answer": "",
    "qury_retry_count": 0
}
final_state = graph.invoke(initial_state)
print("\n✅ Final Answer:")
print(final_state["answer"])