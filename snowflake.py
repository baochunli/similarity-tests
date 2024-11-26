from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")

# queries = ['what is snowflake?', 'Where can I get the best tacos?']
# documents = ['The Data Cloud!', 'Mexico City of Course!']
queries = ["That is a happy person"]
documents = ["That is a happy dog", "That is a very happy person", "Today is a sunny day"]

query_embeddings = model.encode(queries, prompt_name="query", convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)

scores = query_embeddings @ document_embeddings.T
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    # Output passages & scores
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)