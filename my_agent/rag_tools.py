def search_knowledge_base_tool(query: str):
    """Search the project management knowledge base."""
    from my_agent.rag import search_knowledge_base

    return search_knowledge_base(query=query, top_k=3)