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