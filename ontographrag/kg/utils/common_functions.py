import hashlib
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from typing import List
import re
import os
import time
from pathlib import Path
from urllib.parse import urlparse
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_VERTEXAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-005": 768,
    "text-multilingual-embedding-002": 768,
    "gemini-embedding-001": 3072,
}

_HF_EMBEDDING_DEFAULTS_BY_PROFILE = {
    "balanced": "sentence-transformers/all-MiniLM-L6-v2",
    "accuracy": "BAAI/bge-base-en-v1.5",
}


def _resolve_openai_embedding_config():
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    dimensions_env = os.getenv("OPENAI_EMBEDDING_DIMENSION", "").strip()
    kwargs = {"model": model_name}
    if dimensions_env:
        dimension = int(dimensions_env)
        kwargs["dimensions"] = dimension
        return kwargs, dimension

    dimension = _OPENAI_EMBEDDING_DIMENSIONS.get(model_name)
    if dimension is None:
        logging.warning(
            "Unknown OpenAI embedding model '%s'; defaulting dimension to 1536. "
            "Set OPENAI_EMBEDDING_DIMENSION explicitly if needed.",
            model_name,
        )
        dimension = 1536
    return kwargs, dimension


def _resolve_vertexai_embedding_config():
    model_name = os.getenv("VERTEXAI_EMBEDDING_MODEL", "text-embedding-005").strip()
    dimensions_env = os.getenv("VERTEXAI_EMBEDDING_DIMENSION", "").strip()
    kwargs = {"model": model_name}
    if dimensions_env:
        return kwargs, int(dimensions_env)

    dimension = _VERTEXAI_EMBEDDING_DIMENSIONS.get(model_name)
    if dimension is None:
        logging.warning(
            "Unknown Vertex AI embedding model '%s'; defaulting dimension to 768. "
            "Set VERTEXAI_EMBEDDING_DIMENSION explicitly if needed.",
            model_name,
        )
        dimension = 768
    return kwargs, dimension


def _resolve_huggingface_embedding_model(explicit_model_name: str = "") -> str:
    normalized = str(explicit_model_name or "").strip()
    normalized_lower = normalized.lower()
    if normalized and normalized_lower not in {"sentence_transformers", "huggingface"}:
        return normalized

    env_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "").strip()
    if env_model:
        return env_model

    profile = str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    if profile in {"accuracy", "quality", "high_accuracy"}:
        return _HF_EMBEDDING_DEFAULTS_BY_PROFILE["accuracy"]
    return _HF_EMBEDDING_DEFAULTS_BY_PROFILE["balanced"]

def load_embedding_model(embedding_model_name: str):
    """
    Load embedding model based on the model name
    Returns embeddings object and dimension
    """
    normalized = str(embedding_model_name or "").strip()
    normalized_lower = normalized.lower()

    if normalized_lower in {"openai"} or normalized in _OPENAI_EMBEDDING_DIMENSIONS:
        kwargs, dimension = _resolve_openai_embedding_config()
        if normalized in _OPENAI_EMBEDDING_DIMENSIONS:
            kwargs["model"] = normalized
            dimension = _OPENAI_EMBEDDING_DIMENSIONS[normalized]
        embeddings = OpenAIEmbeddings(**kwargs)
        logging.info(
            "Embedding: Using OpenAI Embeddings model=%s, Dimension:%s",
            kwargs["model"],
            dimension,
        )
    elif normalized_lower in {"vertexai"} or normalized in _VERTEXAI_EMBEDDING_DIMENSIONS:
        kwargs, dimension = _resolve_vertexai_embedding_config()
        if normalized in _VERTEXAI_EMBEDDING_DIMENSIONS:
            kwargs["model"] = normalized
            dimension = _VERTEXAI_EMBEDDING_DIMENSIONS[normalized]
        embeddings = VertexAIEmbeddings(**kwargs)
        logging.info(
            "Embedding: Using Vertex AI Embeddings model=%s, Dimension:%s",
            kwargs["model"],
            dimension,
        )
    elif normalized_lower == "titan":
        embeddings = get_bedrock_embeddings()
        dimension = 1536
        logging.info(f"Embedding: Using bedrock titan Embeddings , Dimension:{dimension}")
    else:
        model_name = _resolve_huggingface_embedding_model(normalized)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name#, cache_folder="/embedding_model"
        )
        sample_vector = embeddings.embed_query("embedding readiness probe")
        dimension = len(sample_vector)
        logging.info(
            "Embedding: Using Langchain HuggingFaceEmbeddings model=%s, Dimension:%s",
            model_name,
            dimension,
        )
    return embeddings, dimension

def create_graph_database_connection(uri, userName, password, database):
    """
    Create Neo4j graph database connection
    """
    enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")

    driver_config = {}
    if enable_user_agent:
        driver_config['user_agent'] = os.environ.get('NEO4J_USER_AGENT')

    # LangChain Neo4jGraph with direct username/password parameters
    graph = Neo4jGraph(url=uri, database=database, username=userName, password=password,
                       refresh_schema=False, sanitize=True, driver_config=driver_config)
    return graph

def delete_uploaded_local_file(merged_file_path, file_name):
    """
    Delete uploaded local file
    """
    file_path = Path(merged_file_path)
    if file_path.exists():
        file_path.unlink()
        logging.info(f'file {file_name} deleted successfully')

def create_gcs_bucket_folder_name_hashed(uri, file_name):
    """
    Create GCS bucket folder name with hash
    """
    folder_name = uri + file_name
    folder_name_sha1 = hashlib.sha1(folder_name.encode())
    folder_name_sha1_hashed = folder_name_sha1.hexdigest()
    return folder_name_sha1_hashed

def get_bedrock_embeddings():
    """
    Creates and returns a BedrockEmbeddings object using the specified model name.
    Args:
        model (str): The name of the model to use for embeddings.
    Returns:
        BedrockEmbeddings: An instance of the BedrockEmbeddings class.
    """
    try:
        env_value = os.getenv("BEDROCK_EMBEDDING_MODEL")
        if not env_value:
            raise ValueError("Environment variable 'BEDROCK_EMBEDDING_MODEL' is not set.")
        try:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        except ValueError:
            raise ValueError(
                "Environment variable 'BEDROCK_EMBEDDING_MODEL' is improperly formatted. "
                "Expected format: 'model_name,aws_access_key,aws_secret_key,region_name'."
            )
        bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name.strip(),
                aws_access_key_id=aws_access_key.strip(),
                aws_secret_access_key=aws_secret_key.strip(),
            )
        bedrock_embeddings = BedrockEmbeddings(
            model_id=model_name.strip(),
            client=bedrock_client
        )
        return bedrock_embeddings
    except Exception as e:
        logging.exception("Unexpected error while loading Bedrock embeddings")
        raise
