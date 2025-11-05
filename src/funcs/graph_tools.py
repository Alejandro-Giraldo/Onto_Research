"""
Graph tools for querying the graph

- Graph Analysis
- Graph Profiler
- Graph Proyections
- Graph Embeddings
- Graph Clustering
- Graph Similarity
- Graph Similarity Search
- Graph Similarity Search with Embeddings
- Graph Similarity Search with Embeddings and Clustering



"""



def query_graph(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return result




