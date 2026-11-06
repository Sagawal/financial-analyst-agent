from rag.embeddings import BGEEmbeddings

embedding_model = BGEEmbeddings()

texts = [
    "Apple reported revenue of 391 billion dollars.",
    "Microsoft reported strong cloud revenue growth.",
    "NVIDIA develops GPUs for artificial intelligence."
]

vectors = embedding_model.embed_documents(texts)

for i, vector in enumerate(vectors):
    print(f"Text {i+1}: Vector dimensions = {len(vector)}")

query = "What was Apple's revenue?"

query_vector = embedding_model.embed_query(query)

print("Query vector dimensions:", len(query_vector))