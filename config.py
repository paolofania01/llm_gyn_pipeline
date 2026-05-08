# =============================================================================
# config.py — Parametri globali del progetto
# =============================================================================
# Modifica questo file per cambiare modello, condizione, o percorsi
# senza toccare il codice dei moduli.

from pathlib import Path

# -----------------------------------------------------------------------------
# Percorsi
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

DATA_RAW       = BASE_DIR / "data" / "raw" / "vs ginecologica_1xlsx.xlsx"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "dataset_clean.csv"
RESULTS_DIR    = BASE_DIR / "data" / "results"
PROMPTS_DIR    = BASE_DIR / "prompts"
DOCS_DIR       = BASE_DIR / "docs"

# -----------------------------------------------------------------------------
# Esperimento
# -----------------------------------------------------------------------------
# Condizione A: solo protocollo, nessun prompt strutturato
# Condizione B: protocollo + regole foglio Excel
# Condizione C: prompt completo ottimizzato
CONDIZIONE = "A"

# -----------------------------------------------------------------------------
# Modello LLM
# -----------------------------------------------------------------------------
# Opzioni supportate: "groq", "ollama"
PROVIDER = "groq"

# Modelli Groq:
#   "llama-3.3-70b-versatile"   ← consigliato per i primi test
#   "llama-3.1-8b-instant"      ← più veloce, meno accurato
GROQ_MODEL = "llama-3.3-70b-versatile"

# Modelli Ollama (da scaricare localmente):
#   "llama3.2:3b"
#   "llama3.1:8b"
#   "mistral:7b"
#   "biomistral:7b"
OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# -----------------------------------------------------------------------------
# Parametri di campionamento
# -----------------------------------------------------------------------------
# None = usa tutto il dataset (6068 righe)
# Numero intero = campione utilizzato per test
SAMPLE_SIZE = 100

# Seed per riproducibilità del campione
RANDOM_SEED = 11

# -----------------------------------------------------------------------------
# Parametri API
# -----------------------------------------------------------------------------
BATCH_SIZE   = 1    # numero prescrizioni per chiamata
API_DELAY    = 0.5  # secondi tra una chiamata e l'altra
MAX_RETRIES  = 3    # tentativi in caso di errore
TEMPERATURE  = 0.0  # 0 = deterministico
MAX_TOKENS   = 512  # massimo token nella risposta

# -----------------------------------------------------------------------------
# Scala di distanza tra priorità
# -----------------------------------------------------------------------------
PRIORITY_ORDER   = {"U": 0, "B": 1, "D": 2, "P": 3}
DISTANZA_CRITICA = 2  # oltre questa soglia = errore clinicamente rilevante