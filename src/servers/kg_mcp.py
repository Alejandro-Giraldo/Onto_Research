'''


MCP Servers tool for:

- Standarization Access to Knolwdege Graph data

'''

from fastmcp import FastMCP
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

mcp = FastMCP("Demo For Globant Knowlde Graphs")

@mcp.tool
def query_graph(query) -> int:
    """Query the graph"""
    return "KG Query Results"

if __name__ == "__main__":
    mcp.run()