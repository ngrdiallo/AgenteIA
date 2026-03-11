#!/bin/bash
set -e
cd /home/kid/IAGestioneArte

# Kill precedenti
pkill -f "python3 main.py" 2>/dev/null || true
sleep 1

# Porta libera
if lsof -i :8001 -t &>/dev/null; then
  echo "Port 8001 occupied. Killing..."
  lsof -i :8001 -t | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# Verifica moduli critici
echo "Verifica dipendenze..."
venv/bin/python3 -c "
import sys
modules = ['groq', 'mistralai', 'google.generativeai', 
           'openai', 'httpx', 'fastapi', 'chromadb', 
           'anthropic', 'sentence_transformers']
missing = []
for m in modules:
    try: __import__(m)
    except ImportError: missing.append(m)
if missing:
    print(f'Moduli mancanti: {missing}')
    sys.exit(1)
print('Tutte le dipendenze presenti')
"

# Verifica ChromaDB
venv/bin/python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./storage/chroma')
docs = client.get_collection('documents').count()
print(f'ChromaDB OK: {docs} documenti')
" || { echo "ChromaDB rotto"; exit 1; }

echo "Avvio server..."
exec venv/bin/python3 main.py
