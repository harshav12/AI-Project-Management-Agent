def search_knowledge_base_tool(query: str, user_id: str = None):
    """Search the existing knowledge base and the user's uploaded documents."""
    from my_agent.rag import search_all_documents

    return search_all_documents(
        query=query,
        user_id=user_id,
        top_k=3
    )