class Settings:
    model_path: str = "app/cnn/best_model.pt"  # <-- RUTA ACTUALIZADA
    num_classes: int = 2
    qdrant_url: str = "http://localhost:6333"
    image_size: int = 224
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "radiografias_columna"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_rag: int = 3

def get_settings():
    return Settings()