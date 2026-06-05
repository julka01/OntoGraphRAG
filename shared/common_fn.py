import os
import logging

def load_embedding_model(embedding_model_name: str = None):
    """
    Load the embedding model based on the given name.
    If no name provided, uses environment variable EMBEDDING_PROVIDER.
    Returns a tuple of (embedding_function, embedding_dimension)
    """
    from ontographrag.kg.utils.common_functions import load_embedding_model as _shared_loader

    provider_name = embedding_model_name or os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_function, embedding_dimension = _shared_loader(provider_name)
    logging.info("Using %s embeddings with dimension %s", provider_name, embedding_dimension)
    return embedding_function, embedding_dimension

def wrap_llm_with_model_name(llm):
    '''
    Ensure an LLM-like object exposes a model_name attribute for legacy callers.
    '''
    if not hasattr(llm, 'model_name'):
        model_name = getattr(llm, 'model', 'unknown_model')
        if not isinstance(model_name, str):
            model_name = str(model_name)
        setattr(llm, 'model_name', model_name)
    return llm

def _add_graph_documents_with_merge(graph, graph_document_list):
    """
    Custom implementation of add_graph_documents using MERGE instead of CREATE
    to handle duplicate IDs gracefully
    """
    for doc in graph_document_list:
        # Process nodes
        for node in doc.nodes:
            # Use MERGE to create or update nodes
            node_query = f"""
            MERGE (n:{node.type} {{id: $id}})
            SET n += $properties
            """
            properties = {}
            if hasattr(node, 'properties') and node.properties:
                properties.update(node.properties)
            # Ensure id is always set
            properties["id"] = node.id

            graph.query(node_query, {
                "id": node.id,
                "properties": properties
            })

        # Process relationships
        for rel in doc.relationships:
            # Use MERGE to create or update relationships
            # Sanitize relationship type to replace spaces with underscores for valid Cypher
            sanitized_rel_type = rel.type.replace(' ', '_').replace('-', '_').upper()

            rel_query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{sanitized_rel_type}]->(target)
            SET r += $properties
            """
            properties = {}
            if hasattr(rel, 'properties') and rel.properties:
                properties.update(rel.properties)

            graph.query(rel_query, {
                "source_id": rel.source.id,
                "target_id": rel.target.id,
                "properties": properties
            })
