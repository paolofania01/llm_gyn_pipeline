# =============================================================================
# 02_llm_classifier.py — Classificazione LLM delle prescrizioni ginecologiche
# =============================================================================
# Per ogni riga del dataset pulito, invia il quesito diagnostico a un LLM
# e si ottiene: macrocategoria clinica + priorità suggerita dal protocollo.
#
# Condizioni sperimentali (impostare in config.py):
#   A — solo protocollo regionale
#   B — protocollo + regole foglio Excel
#   C — protocollo + regole foglio Excel + esempi few-shot
#
# Output: data/results/condizione_{CONDIZIONE}/classificazioni_{CONDIZIONE}.csv


import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
 
import pandas as pd
from groq import Groq
from tqdm import tqdm
 
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED, RESULTS_DIR, CONDIZIONE,
    GROQ_MODEL, API_DELAY, MAX_RETRIES, TEMPERATURE, MAX_TOKENS,
    PRIORITY_ORDER, DISTANZA_CRITICA, SAMPLE_SIZE, RANDOM_SEED
)

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Costanti cliniche
# =============================================================================
 
PRIORITA_VALIDE = {"U", "B", "D", "P", "PS", "ND"}
# ND = Non Determinabile (quesito troppo vago o assente)
# PS = il modello ritiene che il caso richieda invio al Pronto Soccorso
 
# =============================================================================
# Testi per i prompt
# =============================================================================
 
PROTOCOLLO_TESTO = """
PROTOCOLLO REGIONALE EMILIA-ROMAGNA — Prima Visita Ginecologica (DM 89.26.1)
 
INVIO PS (Pronto Soccorso):
  - Dolore pelvico acuto
  - Menometrorragie gravi (molto più di una normale perdita mestruale)
  - Sospetto abuso sessuale
 
PRIORITÀ U — Urgente (entro 72 ore):
  - Ascesso della ghiandola di Bartolini
  - Ascite da probabile patologia ginecologica
 
PRIORITÀ B — Breve (entro 10 giorni):
  - Vaginiti persistenti resistenti a terapia (dopo test microbiologico)
  - Pap Test positivo (se non afferente al percorso screening)
  - Perdite ematiche atipiche (escluse menometrorragie gravi) in menopausa
  - Sospetta neoplasia ginecologica
 
PRIORITÀ D — Differibile (entro 30 giorni):
  - Cisti ovarica ≥ 3 cm
  - Dolore pelvico cronico (inclusa sospetta endometriosi)
  - Fibromi uterini sintomatici
  - Irregolarità mestruale non in perimenopausa
  - Dolore vulvare
  - Altre condizioni cliniche non altrimenti specificate
 
PRIORITÀ P — Programmabile (entro 120 giorni):
  - Amenorrea con test di gravidanza negativo
  - Sospetta sindrome dell'ovaio policistico (PCOS)
  - Menopausa sintomatica
  - Sterilità/infertilità di coppia
  - Incontinenza urinaria
  - Prolasso utero-vaginale
  - Altre condizioni cliniche non altrimenti specificate
 
NOTA: La visita ginecologica NON è indicata come screening in donne asintomatiche.
In caso di contraccezione, indirizzare al consultorio (nessuna priorità SSN).
""".strip()
 
REGOLE_EXCEL = """
REGOLE PER LA CLASSIFICAZIONE DEI QUESITI DIAGNOSTICI:
 
Obiettivo:
  1. Creare macrocategorie a partire dai quesiti clinici e dalle indicazioni del protocollo regionale
  2. Associare ogni quesito clinico a una macrocategoria
  3. Valutare la coerenza tra prescrizione e protocollo regionale
 
Regole generali:
  - Considera il contenuto clinico principale
  - Utilizza parole chiave anche abbreviate o con errori ortografici
  - Assegna UNA sola categoria prevalente
  - Se presenti più condizioni cliniche rilevanti → assegna "fattori multipli"
  - Se il quesito è vago o non specifico → assegna "altro clinico/non specificato"
  - Se il quesito è assente → assegna "quesito assente"
 
Parole chiave speciali:
  - controllo, ctr, follow up, fu, accertamenti → "controllo/follow up"
  - 617 (codice ICD) → "endometriosi"
 
Sinonimi e abbreviazioni:
  - menorragia, metrorragia, spotting → sanguinamento uterino anomalo
  - ciclo irreg, amenorrea, oligomenorrea → disturbi del ciclo mestruale
  - dol pelv, algie pelviche → dolore pelvico
  - cisti, formazione annessiale → patologia annessiale
  - fibroma, mioma → patologia uterina benigna
  - hpv, pap test anomalo, ascus, lsil → screening / patologia cervicale
  - gravid, gestaz → gravidanza
  - infertilità, sterilità → infertilità/sterilità
  - perdite vaginali, leucorrea → infezioni / disturbi vaginali
  - dismenorrea → dolore mestruale
  - endometriosi sospetta → endometriosi
  - prolasso, incontinenza → disturbi del pavimento pelvico
""".strip()
 
# =============================================================================
# Prompt builder — tre condizioni sperimentali
# =============================================================================
 
def build_prompt_A(quesito: str) -> str:
    """Condizione A: solo protocollo regionale."""
    return f"""{PROTOCOLLO_TESTO}
 
---
Sei un assistente clinico esperto. Analizza il seguente quesito diagnostico scritto da un medico di base
per una prescrizione di Prima Visita Ginecologica.
 
QUESITO: "{quesito}"
 
Basandoti esclusivamente sul protocollo sopra, rispondi con un oggetto JSON con questi campi:
  "categoria": la macrocategoria clinica prevalente (stringa descrittiva)
  "priorita_suggerita": la priorità raccomandata dal protocollo (U, B, D, P, PS, o ND)
  "motivazione": spiegazione sintetica in 1-2 frasi
 
Rispondi SOLO con il JSON, senza testo aggiuntivo."""
 
 
def build_prompt_B(quesito: str) -> str:
    """Condizione B: protocollo + regole foglio Excel."""
    return f"""{PROTOCOLLO_TESTO}
 
---
{REGOLE_EXCEL}
 
---
Sei un assistente clinico esperto. Analizza il seguente quesito diagnostico scritto da un medico di base
per una prescrizione di Prima Visita Ginecologica.
 
QUESITO: "{quesito}"
 
Rispondi con un oggetto JSON con questi campi:
  "categoria": la macrocategoria clinica prevalente
  "priorita_suggerita": una tra U, B, D, P, PS, ND
  "motivazione": spiegazione sintetica in 1-2 frasi
 
Rispondi SOLO con il JSON, senza testo aggiuntivo."""
 
 
def build_prompt_C(quesito: str) -> str:
    """Condizione C: protocollo + regole foglio Excel + esempi few-shot."""
    esempi = """
        ESEMPI:
 
        Quesito: "perdite ematiche in menopausa"
        {"categoria": "sanguinamento uterino anomalo", "priorita_suggerita": "B", "motivazione": "Perdite ematiche atipiche in menopausa rientrano nella priorità B per il rischio di patologia endometriale."}
 
        Quesito: "cisti ovarica dx 4 cm"
        {"categoria": "patologia annessiale", "priorita_suggerita": "D", "motivazione": "Cisti ovarica ≥ 3 cm richiede visita differibile entro 30 giorni secondo il protocollo."}
 
        Quesito: "controllo ginecologico"
        {"categoria": "controllo/follow up", "priorita_suggerita": "ND", "motivazione": "Controllo generico in donna asintomatica non è indicazione alla visita SSN secondo il protocollo."}
 
        Quesito: "dolore pelvico e irregolarità mestruale"
        {"categoria": "fattori multipli", "priorita_suggerita": "D", "motivazione": "Presenza di due condizioni (dolore pelvico cronico e irregolarità mestruale), entrambe con priorità D."}
 
        Quesito: ""
        {"categoria": "quesito assente", "priorita_suggerita": "ND", "motivazione": "Nessun quesito diagnostico fornito."}
        """.strip()
 
    return f"""{PROTOCOLLO_TESTO}
 
            ---
            {REGOLE_EXCEL}
 
            ---
            {esempi}
 
            ---
            Sei un assistente clinico esperto. Analizza il seguente quesito diagnostico scritto da un medico di base
            per una prescrizione di Prima Visita Ginecologica.
 
            QUESITO: "{quesito}"
 
            Rispondi con un oggetto JSON con questi campi:
                "categoria": la macrocategoria clinica prevalente
                "priorita_suggerita": una tra U, B, D, P, PS, ND
                "motivazione": spiegazione sintetica in 1-2 frasi
 
            Rispondi SOLO con il JSON, senza testo aggiuntivo."""
 
PROMPT_BUILDERS = {
    "A": build_prompt_A,
    "B": build_prompt_B,
    "C": build_prompt_C,
}
 
# =============================================================================
# Parsing output LLM
# =============================================================================
 
def estrai_json(testo: str) -> Optional[dict]:
    """
    Estrae il primo oggetto JSON valido dalla risposta del modello.
    """
    if not testo:
        return None
 
    # Prova 1: risposta già JSON pulito
    try:
        return json.loads(testo.strip())
    except json.JSONDecodeError:
        pass
 
    # Prova 2: cerca blocco ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", testo, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
 
    # Prova 3: cerca il primo { ... } nel testo
    match = re.search(r"\{[^{}]*\}", testo, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
 
    return None
 
 
def valida_output(parsed: dict) -> dict:
    """Valida e normalizza i campi del JSON estratto."""
    # Priorità: normalizza in maiuscolo e verifica
    priorita = str(parsed.get("priorita_suggerita", "ND")).strip().upper()
    if priorita not in PRIORITA_VALIDE:
        # cerca la lettera valida nel testo
        for p in ["PS", "U", "B", "D", "P"]:
            if p in priorita:
                priorita = p
                break
        else:
            priorita = "ND"
    
    categoria = str(parsed.get("categoria", "altro clinico/non specificato")).strip()
    motivazione = str(parsed.get("motivazione", "")).strip()
 
    return {
        "categoria_llm": categoria,
        "priorita_suggerita": priorita,
        "motivazione": motivazione,
        "parse_ok": True,
    }
 
 
def fallback_output(motivo: str = "") -> dict:
    """Output di fallback per righe che non si riescono a classificare."""
    return {
        "categoria_llm": "altro clinico/non specificato",
        "priorita_suggerita": "ND",
        "motivazione": f"[ERRORE PARSING] {motivo}",
        "parse_ok": False,
    }
 
# =============================================================================
# Client Groq
# =============================================================================
 
def chiama_groq(client: Groq, prompt: str) -> Optional[str]:
    """Singola chiamata all'API Groq. Restituisce il testo della risposta o None."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Errore chiamata Groq: {e}")
        return None
 
 
def classifica_quesito(client: Groq, quesito: str, condizione: str) -> dict:
    """
    Classifica un singolo quesito diagnostico.
    Gestisce retry automatico e fallback.
    """
    build_prompt = PROMPT_BUILDERS[condizione]
    prompt = build_prompt(quesito)
 
    for tentativo in range(1, MAX_RETRIES + 1):
        raw = chiama_groq(client, prompt)
 
        if raw is None:
            print(f"  Tentativo {tentativo}/{MAX_RETRIES}: risposta nulla")
            time.sleep(API_DELAY * 2)
            continue
 
        parsed = estrai_json(raw)
 
        if parsed is None:
            print(f"  Tentativo {tentativo}/{MAX_RETRIES}: JSON non trovato")
            time.sleep(API_DELAY)
            continue
 
        return valida_output(parsed)
 
    # Tutti i tentativi falliti
    print(f"  FALLBACK dopo {MAX_RETRIES} tentativi per quesito: '{quesito[:60]}'")
    return fallback_output("Max retry raggiunto")
 
# =============================================================================
# Calcolo distanza priorità e match
# =============================================================================
 
def calcola_distanza(p_medico: str, p_suggerita: str) -> Optional[int]:
    """
    Distanza tra due priorità (U=0, B=1, D=2, P=3).
    Restituisce None se una delle due è PS o ND (non comparabile).
    """
    if p_medico in ("PS", "ND") or p_suggerita in ("PS", "ND"):
        return None
    try:
        return abs(PRIORITY_ORDER[p_medico] - PRIORITY_ORDER[p_suggerita])
    except KeyError:
        return None
 
 
def classifica_match(distanza: Optional[int]) -> str:
    """
    Restituisce una etichetta in base alla distanza tra priorità:
      match          — priorità identiche
      vicino         — distanza 1 (accettabile)
      errore         — distanza 2 (clinicamente rilevante)
      errore_critico — distanza >= DISTANZA_CRITICA
      non_valutabile — PS, ND, o dati mancanti
    """
    if distanza is None:
        return "non_valutabile"
    if distanza == 0:
        return "match"
    if distanza == 1:
        return "vicino"
    if distanza >= DISTANZA_CRITICA:
        return "errore_critico"
    return "errore"
 
# =============================================================================
# Pipeline principale
# =============================================================================
 
def run_classification() -> pd.DataFrame:
    """
    Esegue la classificazione LLM su tutto il dataset (o campione).
    Salva il CSV dei risultati e restituisce il DataFrame.
    """
    print("=" * 60)
    print(f"CLASSIFICAZIONE LLM — Condizione {CONDIZIONE}")
    print(f"Modello: {GROQ_MODEL}")
    print("=" * 60)
 
    # 1. Lettura dataset preprocessato
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Dataset pulito non trovato: {DATA_PROCESSED}\n"
            "Esegui prima: python preprocessing.py"
        )
    df = pd.read_csv(DATA_PROCESSED, encoding="utf-8")
    print(f"\n[1/4] Dataset caricato: {len(df)} righe")
 
    # 2. Campionamento opzionale
    if SAMPLE_SIZE is not None:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"      Campione: {len(df)} righe (seed={RANDOM_SEED})")
 
    # 3. Inizializza client Groq
    print(f"\n[2/4] Inizializzazione client Groq...")
    client = Groq()  # legge GROQ_API_KEY dall'ambiente
    print(f"      OK")
 
    # 4. Loop classificazione
    print(f"\n[3/4] Classificazione in corso...")
    risultati = []
 
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Condizione {CONDIZIONE}"):
        quesito = str(row.get("q_clean", "")).strip()
 
        # Quesiti vuoti: skip LLM, assegna direttamente
        if not quesito:
            res = {
                "categoria_llm": "quesito assente",
                "priorita_suggerita": "ND",
                "motivazione": "Quesito vuoto.",
                "parse_ok": True,
            }
        else:
            res = classifica_quesito(client, quesito, CONDIZIONE)
            time.sleep(API_DELAY)
 
        risultati.append(res)
 
    # 5. Costruisci DataFrame risultati
    df_res = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(risultati).reset_index(drop=True)],
        axis=1
    )
 
    # 6. Confronto priorità
    print(f"\n[4/4] Calcolo confronto priorità...")
    df_res["distanza"] = df_res.apply(
        lambda r: calcola_distanza(str(r["PRIORITA"]), r["priorita_suggerita"]), axis=1
    )
    df_res["match_label"] = df_res["distanza"].apply(classifica_match)
 
    # 7. Salvataggio
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"condizione_{CONDIZIONE}" / f"classificazioni_{CONDIZIONE}.csv"
    df_res.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Risultati salvati in: {output_path}")
 
    # 8. Statistiche rapide
    n_tot = len(df_res)
    n_ok = df_res["parse_ok"].sum()
    print(f"\nParsing riuscito: {n_ok}/{n_tot} ({n_ok/n_tot*100:.1f}%)")
    print("\nDistribuzione match:")
    print(df_res["match_label"].value_counts().to_string())
 
    return df_res
 
# =============================================================================
# Entry point
# =============================================================================
 
if __name__ == "__main__":
    df_out = run_classification()
 
    print("\nAnteprima output:")
    cols = ["PRIORITA", "q_clean", "categoria_llm", "priorita_suggerita", "distanza", "match_label"]
    cols_presenti = [c for c in cols if c in df_out.columns]
    print(df_out[cols_presenti].head(10).to_string())