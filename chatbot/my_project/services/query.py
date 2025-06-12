from models.llm import get_answer_and_themes,model_call
from models.embeddings import semantic_search
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

chat_history = []

def answer_question_with_themes(query):
    top_docs = semantic_search(query)
    return get_answer_and_themes(query, top_docs)
    
def chatbot(query):
    query_answer = answer_question_with_themes(query)
    initial_context = [SystemMessage(content=query_answer)]
    chat_history.append(initial_context)
    chat_history.append([HumanMessage(content=query)])
    model = model_call()
    while True:
        if query.lower() == "exit-bye":
            break
        response = model.invoke(chat_history)
        chat_history.append([AIMessage(content=response.content)])