from elasticsearch import Elasticsearch, exceptions


def create_es_index(
    index_name: str,
    embedding_dim: int,
    es_host: str = "http://localhost:9200",
    text_field: str = "content",
    vector_field: str = "embedding"
) -> bool:
    """
    Creates an Elasticsearch index with mappings optimized for BM25 full-text search on a text field
    and k-NN vector search on an embedding field.

    Parameters:
    - index_name: Name of the Elasticsearch index to create.
    - embedding_dim: Dimension of the embedding vectors.
    - es_host: URL of the Elasticsearch host (default: localhost:9200).
    - text_field: Name of the text field for BM25 search (default: 'content').
    - vector_field: Name of the dense_vector field for k-NN (default: 'embedding').

    Returns:
    - True if index creation succeeded or already exists, False otherwise.
    """
    # Initialize client
    es = Elasticsearch(es_host)

    # Define index mapping
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {
                    "type": "keyword"
                },
                text_field: {
                    "type": "text",
                    # BM25 is the default similarity in ES
                    "similarity": "BM25"
                },
                vector_field: {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    # Optionally configure kNN params if using the k-NN plugin
                    # "index": true,
                    # "similarity": "l2_norm"
                }
            }
        }
    }

    try:
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
            return True

        # Create index
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
        return True

    except exceptions.ElasticsearchException as e:
        print(f"Error creating index '{index_name}': {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Create an index 'documents' with 768-dimensional embeddings
    success = create_es_index("documents", embedding_dim=1024)
    if success:
        print("Ready to index documents!")
