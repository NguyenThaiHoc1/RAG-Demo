from embedding_model.hf import HFEmbeddingModel

model_embedding_huggingface = HFEmbeddingModel(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)