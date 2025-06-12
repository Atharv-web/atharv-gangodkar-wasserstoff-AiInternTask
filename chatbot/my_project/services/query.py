from models.llm import get_answer_and_themes,model_call
from models.embeddings import semantic_search

def answer_question_with_themes(query):
    top_docs = semantic_search(query)
    return get_answer_and_themes(query, top_docs)
    
def chatbot(chat_history):
    model = model_call()
    response = model.invoke(chat_history)
    return response.content