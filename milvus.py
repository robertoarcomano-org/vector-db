"""
Test Milvus Lite — modalità embedded su file locale (zero server)
Installa: pip install pymilvus
"""

from pymilvus import MilvusClient, DataType

# ─────────────────────────────────────────
# 1. CONNESSIONE
# ─────────────────────────────────────────

# Milvus Lite — salva su file locale (zero server)
client = MilvusClient("./mio_milvus.db")

# Oppure in-memory puro:
# client = MilvusClient(":memory:")

# Oppure server remoto (Docker o Zilliz Cloud):
# client = MilvusClient(uri="http://localhost:19530")
# client = MilvusClient(uri="https://xxx.zillizcloud.com", token="...")

print("✓ Client creato")

# ─────────────────────────────────────────
# 2. CREA UNA COLLECTION (≈ tabella)
# ─────────────────────────────────────────

COLLECTION = "documenti"
DIM = 4  # in produzione: 768, 1536, 3072...

# Rimuovi se esiste già (utile in fase di test)
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# Schema esplicito
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("id",       DataType.INT64,        is_primary=True)
schema.add_field("vector",   DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field("testo",    DataType.VARCHAR,       max_length=512)
schema.add_field("categoria",DataType.VARCHAR,       max_length=64)

# Indice per la ricerca vettoriale
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",   # oppure L2, IP (inner product)
    index_type="FLAT",      # FLAT = esatto, IVF_FLAT = approssimato (più veloce su dataset grandi)
)

client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    index_params=index_params,
)

print(f"✓ Collection '{COLLECTION}' creata")

# ─────────────────────────────────────────
# 3. INSERISCI DATI
# ─────────────────────────────────────────

dati = [
    {"id": 1, "vector": [0.10, 0.20, 0.30, 0.40], "testo": "il gatto dorme sul divano",      "categoria": "animali"},
    {"id": 2, "vector": [0.90, 0.80, 0.10, 0.20], "testo": "pizza margherita fatta in casa", "categoria": "cibo"},
    {"id": 3, "vector": [0.50, 0.50, 0.50, 0.50], "testo": "python è un linguaggio versatile","categoria": "tech"},
    {"id": 4, "vector": [0.15, 0.25, 0.35, 0.45], "testo": "il cane corre nel parco",        "categoria": "animali"},
    {"id": 5, "vector": [0.85, 0.75, 0.15, 0.25], "testo": "spaghetti alla carbonara",       "categoria": "cibo"},
]

res = client.insert(collection_name=COLLECTION, data=dati)
print(f"✓ {res['insert_count']} punti inseriti")

# ─────────────────────────────────────────
# 4. RICERCA PER SIMILARITÀ
# ─────────────────────────────────────────

print("\n── Ricerca base (top 3 più simili) ──")

query_vector = [0.12, 0.22, 0.32, 0.42]

risultati = client.search(
    collection_name=COLLECTION,
    data=[query_vector],        # lista di query (puoi farne più di una in batch)
    limit=3,
    output_fields=["testo", "categoria"],
)

for r in risultati[0]:          # risultati[0] = prima query
    print(f"  id={r['id']}  score={r['distance']:.4f}  testo='{r['entity']['testo']}'")

# ─────────────────────────────────────────
# 5. RICERCA CON FILTRO (vector + metadata)
# ─────────────────────────────────────────

print("\n── Ricerca filtrata (solo categoria=animali) ──")

risultati_filtrati = client.search(
    collection_name=COLLECTION,
    data=[query_vector],
    filter='categoria == "animali"',    # sintassi SQL-like
    limit=5,
    output_fields=["testo", "categoria"],
)

for r in risultati_filtrati[0]:
    print(f"  id={r['id']}  score={r['distance']:.4f}  testo='{r['entity']['testo']}'")

# ─────────────────────────────────────────
# 6. QUERY PER METADATA (senza vettore)
# ─────────────────────────────────────────

print("\n── Query solo per metadata (tutti i cibo) ──")

risultati_query = client.query(
    collection_name=COLLECTION,
    filter='categoria == "cibo"',
    output_fields=["id", "testo", "categoria"],
)

for r in risultati_query:
    print(f"  id={r['id']}  testo='{r['testo']}'")

# ─────────────────────────────────────────
# 7. RECUPERA PER ID
# ─────────────────────────────────────────

print("\n── Recupera id=3 ──")

res = client.get(
    collection_name=COLLECTION,
    ids=[3],
    output_fields=["testo", "vector"],
)

print(f"  testo='{res[0]['testo']}'")
print(f"  vector={res[0]['vector']}")

# ─────────────────────────────────────────
# 8. ELIMINA UN PUNTO
# ─────────────────────────────────────────

client.delete(collection_name=COLLECTION, ids=[5])
print("\n── Punto id=5 eliminato ──")

# ─────────────────────────────────────────
# 9. INFO COLLECTION
# ─────────────────────────────────────────

info = client.get_collection_stats(COLLECTION)
print(f"\n── Info collection ──")
print(f"  Punti totali : {info['row_count']}")

desc = client.describe_collection(COLLECTION)
print(f"  Nome         : {desc['collection_name']}")

print("\n✓ Test completato")