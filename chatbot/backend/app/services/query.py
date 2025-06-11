from app.models.llm import get_answer_and_themes
from app.models.embeddings import semantic_search

def answer_question_with_themes(query):
    top_docs = semantic_search(query, k=1)
    return get_answer_and_themes(query, top_docs)