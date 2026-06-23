"""
Test Qdrant — modalità in-memory (zero server, come LanceDB)
Installa: pip install qdrant-client
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# ─────────────────────────────────────────
# 1. CONNESSIONE
# ─────────────────────────────────────────

client = QdrantClient(":memory:")

# Oppure locale su disco:
# client = QdrantClient(path="./mio_qdrant_db")

# Oppure server remoto:
# client = QdrantClient(host="localhost", port=6333)

print("✓ Client creato")

# ─────────────────────────────────────────
# 2. CREA UNA COLLECTION
# ─────────────────────────────────────────

COLLECTION = "documenti"
DIM = 4

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

print(f"✓ Collection '{COLLECTION}' creata")

# ─────────────────────────────────────────
# 3. INSERISCI PUNTI
# ─────────────────────────────────────────

punti = [
    PointStruct(
        id=1,
        vector=[0.10, 0.20, 0.30, 0.40],
        payload={"testo": "il gatto dorme sul divano", "categoria": "animali"},
    ),
    PointStruct(
        id=2,
        vector=[0.90, 0.80, 0.10, 0.20],
        payload={"testo": "pizza margherita fatta in casa", "categoria": "cibo"},
    ),
    PointStruct(
        id=3,
        vector=[0.50, 0.50, 0.50, 0.50],
        payload={"testo": "python è un linguaggio versatile", "categoria": "tech"},
    ),
    PointStruct(
        id=4,
        vector=[0.15, 0.25, 0.35, 0.45],
        payload={"testo": "il cane corre nel parco", "categoria": "animali"},
    ),
    PointStruct(
        id=5,
        vector=[0.85, 0.75, 0.15, 0.25],
        payload={"testo": "spaghetti alla carbonara", "categoria": "cibo"},
    ),
]

client.upsert(collection_name=COLLECTION, points=punti)
print(f"✓ {len(punti)} punti inseriti")

# ─────────────────────────────────────────
# 4. RICERCA PER SIMILARITÀ
# ─────────────────────────────────────────

print("\n── Ricerca base (top 3 più simili) ──")

query_vector = [0.12, 0.22, 0.32, 0.42]

risultati = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=3,
).points

for r in risultati:
    print(f"  id={r.id}  score={r.score:.4f}  testo='{r.payload['testo']}'")

# ─────────────────────────────────────────
# 5. RICERCA CON FILTRO
# ─────────────────────────────────────────

print("\n── Ricerca filtrata (solo categoria=animali) ──")

risultati_filtrati = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="categoria",
                match=MatchValue(value="animali"),
            )
        ]
    ),
    limit=5,
).points

for r in risultati_filtrati:
    print(f"  id={r.id}  score={r.score:.4f}  testo='{r.payload['testo']}'")

# ─────────────────────────────────────────
# 6. RECUPERA UN PUNTO PER ID
# ─────────────────────────────────────────

print("\n── Recupera punto id=3 ──")

punti_recuperati = client.retrieve(
    collection_name=COLLECTION,
    ids=[3],
    with_payload=True,
    with_vectors=True,
)

p = punti_recuperati[0]
print(f"  testo='{p.payload['testo']}'")
print(f"  vector={p.vector}")

# ─────────────────────────────────────────
# 7. AGGIORNA PAYLOAD
# ─────────────────────────────────────────

client.set_payload(
    collection_name=COLLECTION,
    payload={"categoria": "animali_domestici"},
    points=[1],
)
print("\n── Payload aggiornato per id=1 ──")

# ─────────────────────────────────────────
# 8. ELIMINA UN PUNTO
# ─────────────────────────────────────────

client.delete(
    collection_name=COLLECTION,
    points_selector=[5],
)
print("── Punto id=5 eliminato ──")

# ─────────────────────────────────────────
# 9. INFO COLLECTION
# ─────────────────────────────────────────

info = client.get_collection(COLLECTION)
print(f"\n── Info collection ──")
print(f"  Punti totali : {info.points_count}")
print(f"  Dimensione   : {info.config.params.vectors.size}")
print(f"  Distanza     : {info.config.params.vectors.distance}")

print("\n✓ Test completato")