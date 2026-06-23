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

## Weaviate
### Code
```
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
```
### Output
```
(.venv) berto@laptop:~/src/vector-db$ python weaviate-db.py 
python: can't open file '/home/berto/src/vector-db/weaviate-db.py': [Errno 2] No such file or directory
(.venv) berto@laptop:~/src/vector-db$ python weaviate-test.py 
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"warning","log_level_env":"","msg":"log level not recognized, defaulting to info","time":"2026-06-23T14:41:52+02:00"}
{"action":"startup","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"Feature flag LD integration disabled: could not locate WEAVIATE_LD_API_KEY env variable","time":"2026-06-23T14:41:52+02:00"}
{"action":"startup","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","default_vectorizer_module":"none","level":"info","msg":"the default vectorizer modules is set to \"none\", as a result all new schema classes without an explicit vectorizer setting, will use this vectorizer","time":"2026-06-23T14:41:52+02:00"}
{"action":"startup","auto_schema_enabled":{},"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"auto schema enabled setting is set to \"\u0026{\u003cnil\u003e {{{} {0 0}} 0 0 {{} 0} {{} 0}} true}\"","time":"2026-06-23T14:41:52+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"No resource limits set, weaviate will use all available memory and CPU. To limit resources, set LIMIT_RESOURCES=true","time":"2026-06-23T14:41:52+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"module offload-s3 is enabled","time":"2026-06-23T14:41:52+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","flag_key":"collection-retrieval-strategy","level":"info","msg":"feature flag instantiated","time":"2026-06-23T14:41:53+02:00","tool":"feature_flag","value":"LeaderOnly"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"warning","msg":"Multiple vector spaces are present, GraphQL Explore and REST API list objects endpoint module include params has been disabled as a result.","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"open cluster service","servers":{"Embedded_at_8079":50189},"time":"2026-06-23T14:41:53+02:00"}
{"address":"192.168.1.8:50190","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"starting cloud rpc server ...","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"starting raft sub-system ...","time":"2026-06-23T14:41:53+02:00"}
{"address":"192.168.1.8:50189","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"tcp transport","tcpMaxPool":3,"tcpTimeout":10000000000,"time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"loading local db","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"local DB successfully loaded","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"schema manager loaded","n":0,"time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","metadata_only_voters":false,"msg":"construct a new raft node","name":"Embedded_at_8079","time":"2026-06-23T14:41:53+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","index":4,"level":"info","msg":"initial configuration","servers":"[[{Suffrage:Voter ID:Embedded_at_8079 Address:192.168.1.8:49609}]]","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","last_snapshot_index":0,"last_store_applied_index_on_start":0,"level":"info","msg":"raft node constructed","raft_applied_index":0,"raft_last_index":4,"time":"2026-06-23T14:41:53+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","follower":{},"leader-address":"","leader-id":"","level":"info","msg":"entering follower state","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","hasState":true,"level":"info","msg":"raft init","time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"attempting to join","remoteNodes":{"Embedded_at_8079":"192.168.1.8:50189"},"time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"attempted to join and failed","remoteNode":"192.168.1.8:50189","status":8,"time":"2026-06-23T14:41:53+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"attempting to join","remoteNodes":{"Embedded_at_8079":"192.168.1.8:50189"},"time":"2026-06-23T14:41:54+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"attempted to join and failed","remoteNode":"192.168.1.8:50189","status":8,"time":"2026-06-23T14:41:54+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","last-leader-addr":"","last-leader-id":"","level":"warning","msg":"heartbeat timeout reached, starting election","time":"2026-06-23T14:41:54+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"entering candidate state","node":{},"term":4,"time":"2026-06-23T14:41:54+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"pre-vote successful, starting election","refused":0,"tally":1,"term":4,"time":"2026-06-23T14:41:54+02:00","votesNeeded":1}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"election won","tally":1,"term":4,"time":"2026-06-23T14:41:54+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","leader":{},"level":"info","msg":"entering leader state","time":"2026-06-23T14:41:54+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"configured versions","server_version":"1.30.5","time":"2026-06-23T14:41:55+02:00","version":"1.30.5"}
{"action":"grpc_startup","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"grpc server listening at [::]:50050","time":"2026-06-23T14:41:55+02:00"}
{"address":"192.168.1.8:50189","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"current Leader","time":"2026-06-23T14:41:55+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"starting migration from old schema","time":"2026-06-23T14:41:55+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"legacy schema is empty, nothing to migrate","time":"2026-06-23T14:41:55+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"migration from the old schema has been successfully completed","time":"2026-06-23T14:41:55+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"attempting to join","remoteNodes":{"Embedded_at_8079":"192.168.1.8:50189"},"time":"2026-06-23T14:41:55+02:00"}
{"action":"raft","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","command":0,"level":"info","msg":"updating configuration","server-addr":"192.168.1.8:50189","server-id":"Embedded_at_8079","servers":"[[{Suffrage:Voter ID:Embedded_at_8079 Address:192.168.1.8:50189}]]","time":"2026-06-23T14:41:55+02:00"}
{"action":"restapi_management","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"Serving weaviate at http://127.0.0.1:8079","time":"2026-06-23T14:41:55+02:00","version":"1.30.5"}
{"action":"telemetry_push","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"telemetry started","payload":"\u0026{MachineID:674b5ca8-1c46-4611-af5a-8d9282eeccfa Type:INIT Version:1.30.5 ObjectsCount:0 OS:linux Arch:amd64 UsedModules:[] CollectionsCount:0}","time":"2026-06-23T14:41:55+02:00"}
✓ Client connesso
/home/berto/src/vector-db/.venv/lib/python3.13/site-packages/weaviate/warnings.py:196: DeprecationWarning: Dep024: You are using the `vectorizer_config` argument in `collection.config.create()`, which is deprecated.
            Use the `vector_config` argument instead.
            
  warnings.warn(
/home/berto/src/vector-db/.venv/lib/python3.13/site-packages/weaviate/warnings.py:206: DeprecationWarning: Dep025: You are using the `vector_index_config` argument in `collection.config.create()`, which is deprecated.
            Use the `vector_config` argument instead defining `vector_index_config` as a sub-argument.
            
  warnings.warn(
✓ Collection 'Documenti' creata
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"warning","msg":"prop len tracker file /home/berto/.local/share/weaviate/documenti/1tXSfVENMtkZ/proplengths does not exist, creating new tracker","time":"2026-06-23T14:41:56+02:00"}
{"action":"hnsw_prefill_cache_async","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"not waiting for vector cache prefill, running in background","time":"2026-06-23T14:41:56+02:00","wait_for_cache_prefill":false}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"Created shard documenti_1tXSfVENMtkZ in 1.468083ms","time":"2026-06-23T14:41:56+02:00"}
{"action":"hnsw_vector_cache_prefill","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","count":1000,"index_id":"main","level":"info","limit":1000000000000,"msg":"prefilled vector cache","time":"2026-06-23T14:41:56+02:00","took":113818}
✓ 5 oggetti inseriti

── Ricerca base (top 3 più simili) ──
  score=0.0004  testo='il gatto dorme sul divano'
  score=0.0006  testo='il cane corre nel parco'
  score=0.0761  testo='python è un linguaggio versatile'

── Ricerca filtrata (solo categoria=animali) ──
  score=0.0004  testo='il gatto dorme sul divano'
  score=0.0006  testo='il cane corre nel parco'

── Query solo metadata (tutti i cibo) ──
  testo='spaghetti alla carbonara'
  testo='pizza margherita fatta in casa'

── Ricerca BM25 (keyword) ──
  bm25_score=1.2170  testo='il gatto dorme sul divano'

── Ricerca ibrida (near_vector + BM25) ──
  hybrid_score=1.0000  testo='il gatto dorme sul divano'
  hybrid_score=0.4997  testo='il cane corre nel parco'
  hybrid_score=0.4141  testo='python è un linguaggio versatile'

── Info collection ──
  Oggetti totali : 5

✓ Test completato
{"action":"restapi_management","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"Shutting down... ","time":"2026-06-23T14:41:56+02:00","version":"1.30.5"}
{"action":"restapi_management","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"Stopped serving weaviate at http://127.0.0.1:8079","time":"2026-06-23T14:41:56+02:00","version":"1.30.5"}
{"action":"telemetry_push","build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"telemetry terminated","payload":"\u0026{MachineID:674b5ca8-1c46-4611-af5a-8d9282eeccfa Type:TERMINATE Version:1.30.5 ObjectsCount:0 OS:linux Arch:amd64 UsedModules:[] CollectionsCount:1}","time":"2026-06-23T14:41:56+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing raft FSM store ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"shutting down raft sub-system ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"transferring leadership to another server","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","error":"cannot find peer","level":"error","msg":"transferring leadership","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing raft-net ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing log store ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing data store ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing loaded database ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing raft-rpc client ...","time":"2026-06-23T14:42:00+02:00"}
{"build_git_commit":"","build_go_version":"go1.24.3","build_image_tag":"","build_wv_version":"1.30.5","level":"info","msg":"closing raft-rpc server ...","time":"2026-06-23T14:42:00+02:00"}
```