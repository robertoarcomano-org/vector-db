"""
Test Weaviate embedded — versione con connect() esplicito
Installa: pip install weaviate-client
"""

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import Filter

# ─────────────────────────────────────────
# 1. CONNESSIONE con connect() esplicito
# ─────────────────────────────────────────

client = weaviate.connect_to_embedded()
client.connect()

print("✓ Client connesso")

try:
    COLLECTION = "Documenti"

    # ─────────────────────────────────────
    # 2. CREA COLLECTION
    # ─────────────────────────────────────

    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)

    collection = client.collections.create(
        name=COLLECTION,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            Property(name="testo",     data_type=DataType.TEXT),
            Property(name="categoria", data_type=DataType.TEXT),
        ],
    )

    print(f"✓ Collection '{COLLECTION}' creata")

    # ─────────────────────────────────────
    # 3. INSERISCI OGGETTI
    # ─────────────────────────────────────

    dati = [
        {"testo": "il gatto dorme sul divano",        "categoria": "animali", "vector": [0.10, 0.20, 0.30, 0.40]},
        {"testo": "pizza margherita fatta in casa",   "categoria": "cibo",    "vector": [0.90, 0.80, 0.10, 0.20]},
        {"testo": "python è un linguaggio versatile", "categoria": "tech",    "vector": [0.50, 0.50, 0.50, 0.50]},
        {"testo": "il cane corre nel parco",          "categoria": "animali", "vector": [0.15, 0.25, 0.35, 0.45]},
        {"testo": "spaghetti alla carbonara",         "categoria": "cibo",    "vector": [0.85, 0.75, 0.15, 0.25]},
    ]

    with collection.batch.dynamic() as batch:
        for d in dati:
            batch.add_object(
                properties={"testo": d["testo"], "categoria": d["categoria"]},
                vector=d["vector"],
            )

    print(f"✓ {len(dati)} oggetti inseriti")

    # ─────────────────────────────────────
    # 4. RICERCA PER SIMILARITÀ
    # ─────────────────────────────────────

    print("\n── Ricerca base (top 3 più simili) ──")

    query_vector = [0.12, 0.22, 0.32, 0.42]

    risultati = collection.query.near_vector(
        near_vector=query_vector,
        limit=3,
        return_properties=["testo", "categoria"],
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    for r in risultati.objects:
        print(f"  score={r.metadata.distance:.4f}  testo='{r.properties['testo']}'")

    # ─────────────────────────────────────
    # 5. RICERCA CON FILTRO
    # ─────────────────────────────────────

    print("\n── Ricerca filtrata (solo categoria=animali) ──")

    risultati_filtrati = collection.query.near_vector(
        near_vector=query_vector,
        filters=Filter.by_property("categoria").equal("animali"),
        limit=5,
        return_properties=["testo", "categoria"],
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    for r in risultati_filtrati.objects:
        print(f"  score={r.metadata.distance:.4f}  testo='{r.properties['testo']}'")

    # ─────────────────────────────────────
    # 6. QUERY SOLO METADATA
    # ─────────────────────────────────────

    print("\n── Query solo metadata (tutti i cibo) ──")

    risultati_query = collection.query.fetch_objects(
        filters=Filter.by_property("categoria").equal("cibo"),
        return_properties=["testo", "categoria"],
    )

    for r in risultati_query.objects:
        print(f"  testo='{r.properties['testo']}'")

    # ─────────────────────────────────────
    # 7. RICERCA BM25 (keyword)
    # ─────────────────────────────────────

    print("\n── Ricerca BM25 (keyword) ──")

    risultati_bm25 = collection.query.bm25(
        query="gatto divano",
        limit=3,
        return_properties=["testo", "categoria"],
        return_metadata=wvc.query.MetadataQuery(score=True),
    )

    for r in risultati_bm25.objects:
        print(f"  bm25_score={r.metadata.score:.4f}  testo='{r.properties['testo']}'")

    # ─────────────────────────────────────
    # 8. RICERCA IBRIDA
    # ─────────────────────────────────────

    print("\n── Ricerca ibrida (near_vector + BM25) ──")

    risultati_ibridi = collection.query.hybrid(
        query="gatto",
        vector=query_vector,
        alpha=0.5,
        limit=3,
        return_properties=["testo", "categoria"],
        return_metadata=wvc.query.MetadataQuery(score=True),
    )

    for r in risultati_ibridi.objects:
        print(f"  hybrid_score={r.metadata.score:.4f}  testo='{r.properties['testo']}'")

    # ─────────────────────────────────────
    # 9. INFO COLLECTION
    # ─────────────────────────────────────

    info = collection.aggregate.over_all(total_count=True)
    print(f"\n── Info collection ──")
    print(f"  Oggetti totali : {info.total_count}")

    print("\n✓ Test completato")

finally:
    client.close()