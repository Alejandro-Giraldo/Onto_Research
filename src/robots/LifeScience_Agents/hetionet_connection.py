"""
Hetionet Neo4j connection utilities.

Separates graph connection concerns from agent logic.


hetionet_url = "https://neo4j.het.io/"
hetionet_username = "neo4j"
hetionet_password = "neo4j"

"""

import os
from typing import Optional
from langchain_neo4j import Neo4jGraph


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable with an optional default."""
    value = os.getenv(name)
    return value if value is not None else default


def create_hetionet_graph(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[Neo4jGraph]:
    """
    Create and return a Neo4jGraph connection to Hetionet.

    Values can be provided explicitly or via environment variables:
    - NEO4J_URI (default: neo4j+s://neo4j.het.io)
    - NEO4J_USERNAME (default: neo4j)
    - NEO4J_PASSWORD (default: neo4j)
    """
    uri = uri or get_env("NEO4J_URI", "neo4j+s://neo4j.het.io")
    username = username or get_env("NEO4J_USERNAME", "neo4j")
    password = password or get_env("NEO4J_PASSWORD", "neo4j")

    try:
        graph = Neo4jGraph(url=uri, username=username, password=password)
        # Ensure schema is loaded for downstream prompting
        try:
            graph.refresh_schema()
        except Exception:
            pass
        print("Successfully connected to Hetionet")
        return graph
    except Exception as exc:
        print(f"Warning: Could not connect to Hetionet: {exc}")
        return None


