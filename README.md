# vector-db
Vector db

## LanceDB
### Code
```
import lancedb
import numpy as np

# Apri (o crea) un database locale
db = lancedb.connect("./mio_db")

# Crea una tabella con dati vettoriali
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "testo": "ciao mondo"},
    {"id": 2, "vector": [0.9, 0.8, 0.1, 0.2], "testo": "hello world"},
    {"id": 3, "vector": [0.5, 0.5, 0.5, 0.5], "testo": "foo bar"},
]

table = db.create_table("mia_tabella", data=data)

# Ricerca per similarità vettoriale
risultati = table.search([0.1, 0.2, 0.3, 0.4]).limit(2).to_pandas()
print(risultati)
```

### Output
```
(.venv) berto@laptop:~/src/vector-db$ python lance-db.py 
   id                vector       testo  _distance
0   1  [0.1, 0.2, 0.3, 0.4]  ciao mondo        0.0
1   3  [0.5, 0.5, 0.5, 0.5]     foo bar        0.3
```

## Qdrant
### Code
```
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
```

### Output
```
(.venv) berto@laptop:~/src/vector-db$ python qdrant.py 
✓ Client creato
✓ Collection 'documenti' creata
✓ 5 punti inseriti

── Ricerca base (top 3 più simili) ──
  id=1  score=0.9996  testo='il gatto dorme sul divano'
  id=4  score=0.9994  testo='il cane corre nel parco'
  id=3  score=0.9239  testo='python è un linguaggio versatile'

── Ricerca filtrata (solo categoria=animali) ──
  id=1  score=0.9996  testo='il gatto dorme sul divano'
  id=4  score=0.9994  testo='il cane corre nel parco'

── Recupera punto id=3 ──
  testo='python è un linguaggio versatile'
  vector=[0.5, 0.5, 0.5, 0.5]

── Payload aggiornato per id=1 ──
── Punto id=5 eliminato ──

── Info collection ──
  Punti totali : 4
  Dimensione   : 4
  Distanza     : Cosine

✓ Test completato
```

## Milvus
## Code
```
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
```
## Output
```
(.venv) berto@laptop:~/src/vector-db$ /home/berto/src/vector-db/.venv/bin/python /home/berto/src/vector-db/milvus.py
✓ Client creato
✓ Collection 'documenti' creata
✓ 5 punti inseriti

── Ricerca base (top 3 più simili) ──
  id=1  score=0.0004  testo='il gatto dorme sul divano'
  id=4  score=0.0006  testo='il cane corre nel parco'
  id=3  score=0.0761  testo='python è un linguaggio versatile'

── Ricerca filtrata (solo categoria=animali) ──
  id=1  score=0.0004  testo='il gatto dorme sul divano'
  id=4  score=0.0006  testo='il cane corre nel parco'

── Query solo per metadata (tutti i cibo) ──
  id=2  testo='pizza margherita fatta in casa'
  id=5  testo='spaghetti alla carbonara'

── Recupera id=3 ──
  testo='python è un linguaggio versatile'
  vector=[0.5, 0.5, 0.5, 0.5]

── Punto id=5 eliminato ──

── Info collection ──
  Punti totali : 4
  Nome         : documenti

✓ Test completato
```