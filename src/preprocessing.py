# =============================================================================
# 01_preprocessing.py — Pulizia e preparazione del dataset
# =============================================================================
# Legge il file Excel originale, pulisce i quesiti diagnostici,
# assegna il vagueness_label, espande le abbreviazioni,
# e salva un CSV pulito pronto per il modulo LLM.
#
# Output: data/processed/dataset_clean.csv

import pandas as pd
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, RANDOM_SEED, SAMPLE_SIZE

# =============================================================================
# Funzioni
# =============================================================================

def normalizza(testo) -> str:
    """Pulizia base del testo."""
    if not isinstance(testo, str):
        return ""
    t = testo.strip().lower()
    t = re.sub(r"[\s\-\.]+$", "", t)   # rimuove " -", ".", spazi finali
    t =re.sub(r'^[ "]+|[ "]+$', '', t) # rimuove spazi e virgolette iniziali/finali
    t = re.sub(r"\s+", " ", t)         # collassa spazi multipli
    t = t.replace("\n", " ").strip()
    return t

# =============================================================================
# Pipeline principale
# =============================================================================

def run_preprocessing(sample_size=None, random_seed=42) -> pd.DataFrame:
    """
    Esegue il preprocessing e salva il CSV pulito.
    Ritorna il DataFrame risultante.
    """
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # 1. Lettura file Excel
    print(f"\n[1/3] Lettura file: {DATA_RAW}")
    df = pd.read_excel(DATA_RAW, sheet_name="data set")
    print(f"      Righe caricate: {len(df)}")
    df = df.rename(columns={"QUESITO_DIAGNOSTICO_INDICE": "QUESITO_ORIGINALE"})

    # 2. Pulizia del dataset
    print("\n[2/3] Normalizzazione testo...")
    df["q_clean"] = df["QUESITO_ORIGINALE"].apply(normalizza)

    # Statistiche di base
    n_vuoti = (df["q_clean"] == "").sum()
    print(f"      Quesiti vuoti: {n_vuoti} ({n_vuoti/len(df)*100:.1f}%)")
    print(f"      Quesiti con testo: {len(df) - n_vuoti} ({(len(df)-n_vuoti)/len(df)*100:.1f}%)")

    # 3. Campionamento stratificato (opzionale)
    if sample_size is not None:
        print(f"\n[3/3] Campionamento stratificato: {sample_size} righe...")
        frame_totale = len(df)
        campioni = []
        for priorita, gruppo in df.groupby("PRIORITA"):
            n = max(1, int(sample_size * len(gruppo) / frame_totale))
            n = min(n, len(gruppo))
            campioni.append(gruppo.sample(n, random_state=random_seed))
        df = pd.concat(campioni).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        print(f"      Righe nel campione: {len(df)}")
        print(f"      Distribuzione priorità: {df['PRIORITA'].value_counts().to_dict()}")
    else:
        print(f"\n[3/3] Nessun campionamento, uso tutto il dataset.")

    # 4. Salvataggio
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False, encoding="utf-8")
    print(f"\n✓ CSV salvato in: {DATA_PROCESSED}")
    print(f"  Colonne: {list(df.columns)}")
    return df


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    df = run_preprocessing(
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )
    print("\nAnteprima:")
    cols = ["PRIORITA", "QUESITO_ORIGINALE", "q_clean"]
    print(df[cols].head(10).to_string())