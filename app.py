import streamlit as st
import pandas as pd
# Versión 2.0 - Fix OpenAI proxies error
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from metrics import wcag_like_contrast_ratio, edge_density, fleiss_kappa_from_long, avg_cohen_kappa_pairwise
import random
import os, unicodedata
from uuid import uuid4
from datetime import date
from pathlib import Path
import json, random, time
from zipfile import ZipFile, is_zipfile
import shutil
import math  
from services.image_generator import generar_imagen
from io import BytesIO

from services.supabase_client import supabase

# Imports de OpenAI (con manejo de errores)
try:
    from services.openai_eval import evaluar_imagen_prompt
    OPENAI_EVAL_AVAILABLE = True
except ImportError:
    OPENAI_EVAL_AVAILABLE = False
    evaluar_imagen_prompt = None

try:
    from config.llm import LLM_ENABLED, is_llm_ready
except ImportError:
    LLM_ENABLED = False
    is_llm_ready = lambda: False
import csv
import base64
ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)  # pon tu imagen en assets/hero_gen-edviz.png
HERO_IMG = ASSETS / "hero_gen-edviz.png"
IA_CSV = Path("data/ia_evaluations/ia_eval.csv")
IA_CSV.parent.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent
CATALOGO_CSV = BASE_DIR / "data" / "catalogo.csv"



import re
from difflib import SequenceMatcher
try:
    from rapidfuzz import fuzz as rf_fuzz   # opcional
except Exception:
    rf_fuzz = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="GEN-EDViz Monitor", layout="wide", initial_sidebar_state="collapsed")

# Cargar configuración LLM desde ENV y, solo si existe, desde secrets.toml
from pathlib import Path

def _safe_load_secrets_to_env():
    KEYS = ("LLM_ENABLE","LLM_PROVIDER","OPENAI_API_KEY","LLM_MODEL","OLLAMA_URL","OLLAMA_MODEL")

    # 1) Si ya están en variables de entorno, no hacemos nada más
    if all(os.getenv(k) for k in KEYS):
        return

    # 2) Sólo intentamos leer st.secrets si realmente hay un secrets.toml presente
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    proj_secrets = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if user_secrets.exists() or proj_secrets.exists():
        try:
            for k in KEYS:
                if k in st.secrets and not os.getenv(k):
                    os.environ[k] = str(st.secrets[k])
        except Exception:
            # Si algo falla, seguimos sin romper la app
            pass

_safe_load_secrets_to_env()

def evaluar_imagen_prompt_simulada(tema, concepto, prompt):
    """
    Simula la evaluación IA de una imagen + prompt.
    Luego esta función se reemplaza por GPT-4o.
    """

    errores_posibles = [
        "No se detectan errores conceptuales.",
        "La estructura no representa correctamente el orden de recorrido.",
        "La imagen puede inducir confusión en nodos visitados.",
        "El concepto se presenta de forma estática cuando es dinámico."
    ]

    recomendaciones_posibles = [
        "Resaltar el nodo actual en cada paso del algoritmo.",
        "Usar flechas direccionales más claras.",
        "Agregar numeración de pasos del algoritmo.",
        "Reducir elementos visuales distractores."
    ]

    resultado = {
        "coherencia": random.randint(3, 5),
        "fidelidad": random.randint(3, 5),
        "claridad": random.randint(2, 5),
        "errores": random.choice(errores_posibles),
        "recomendaciones": random.choice(recomendaciones_posibles),
        "fecha": datetime.now().isoformat()
    }

    return resultado


def extraer_puntaje(valor):
    if pd.isna(valor):
        return None
    
    # Buscar patrón tipo "3/5"
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", str(valor))
    if match:
        return float(match.group(1))
    
    # Si ya es número limpio
    try:
        return float(valor)
    except:
        return None

def guardar_eval_ia(row, resultado):
    existe = IA_CSV.exists()

    with open(IA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow([
                "image_id", "tema", "concepto", "prompt",
                "coherencia", "fidelidad", "claridad",
                "errores", "recomendaciones", "fecha"
            ])

        writer.writerow([
            row["image_id"],
            row["tema"],
            row["concepto"],
            row["prompt"],
            resultado["coherencia"],
            resultado["fidelidad"],
            resultado["claridad"],
            resultado["errores"],
            resultado["recomendaciones"],
            resultado["fecha"]
        ])

def rerun():
    # Compatible con versiones nuevas y antiguas
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _safe_path(v) -> str:
        """
        Devuelve '' si v es None / NaN / 'nan'.
        Si viene algo válido, devuelve str(v). No valida existencia en disco.
        """
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        s = str(v).strip()
        return "" if s.lower() == "nan" else s

# ==== HELPERS para generación automática de preguntas/distractores ====
def _norm_txt(s: str) -> str:
    s = (s or "")
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _ensure_unique_options(opts: list[str]) -> list[str]:
    out, seen = [], set()
    for o in opts:
        o2 = _norm_txt(o)
        if not o2:
            continue
        k = o2.lower()
        if k in seen:
            continue
        out.append(o2)
        seen.add(k)
    return out

def _tfidf_distractors(correct: str,
                       corpus_prompts: list[str],
                       corpus_ids: list[str] | None = None,
                       k: int = 3) -> list[str]:
    """
    Devuelve hasta k prompts de corpus similares al 'correct' (sin repetir el correcto).
    """
    correct_n = _norm_txt(correct)
    if not correct_n:
        return []
    # Construimos el corpus con el correcto como query
    base = [correct_n] + [ _norm_txt(x) for x in corpus_prompts ]
    try:
        X = TfidfVectorizer(min_df=1).fit_transform(base)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()
        order = sims.argsort()[::-1]
    except Exception:
        # fallback por si hay algún problema con TF-IDF
        order = list(range(len(corpus_prompts)))

    out, seen = [], set()
    for idx in order:
        cand = _norm_txt(corpus_prompts[idx])
        if not cand:
            continue
        if cand.lower() == correct_n.lower():
            continue
        if cand.lower() in seen:
            continue
        out.append(cand)
        seen.add(cand.lower())
        if len(out) >= k:
            break
    return out

def cargar_catalogo():
    if CATALOGO_CSV.exists():
        return pd.read_csv(CATALOGO_CSV, encoding="utf-8")
    else:
        return pd.DataFrame()

def _mutate_prompt(p: str) -> str:
    """
    Pequeñas mutaciones verosímiles (para respaldo).
    Cambia términos comunes en ED o adjetivos sin destruir el sentido general.
    """
    p = _norm_txt(p)
    if not p:
        return ""
    # Cambios típicos en ED
    swaps = [
        (r"\bBFS\b", "DFS"), (r"\bDFS\b", "BFS"),
        (r"\bmin-heap\b", "max-heap"), (r"\bmax-heap\b", "min-heap"),
        (r"\bárbol\b", "grafo"), (r"\bgrafo\b", "árbol"),
        (r"\bcola(s)?\b", "pila"), (r"\bpila(s)?\b", "cola"),
        (r"\bdirigido\b", "no dirigido"), (r"\bno dirigido\b", "dirigido"),
    ]
    for pat, rep in swaps:
        if re.search(pat, p, flags=re.I):
            return re.sub(pat, rep, p, count=1, flags=re.I)

    # Cambios de “tono”
    alt = {
        "claro": "detallado",
        "simple": "complejo",
        "detallado": "simple",
        "esquemático": "realista",
        "realista": "esquemático",
    }
    for k, v in alt.items():
        if re.search(rf"\b{k}\b", p, flags=re.I):
            return re.sub(rf"\b{k}\b", v, p, count=1, flags=re.I)

    # Fallback: reordenar ligeramente una frase final (muy simple)
    return p + " con variaciones menores"

def _pick_concept_distractors(correct: str, k: int = 3) -> list[str]:
    """
    Elige conceptos similares del catálogo (mismo dominio) que no sean el correcto.
    Si hay rapidfuzz, prioriza por similitud; si no, usa SequenceMatcher.
    """
    pool = df_meta.get("concepto", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
    pool = [_norm_txt(x) for x in pool if _norm_txt(x)]
    corr = _norm_txt(correct).lower()
    pool = [x for x in pool if x.lower() != corr]
    if not pool:
        return []

    def _score(x: str) -> float:
        a, b = x.lower(), corr
        if rf_fuzz:
            return float(rf_fuzz.ratio(a, b))
        else:
            return SequenceMatcher(None, a, b).ratio() * 100.0

    scored = sorted(pool, key=_score, reverse=True)
    out, seen = [], set()
    for s in scored:
        if s.lower() in seen: 
            continue
        out.append(s)
        seen.add(s.lower())
        if len(out) >= k:
            break
    return out

# ➜ coloca este bloque junto a tus helpers de query params
def _qp_get():
    try:    return dict(st.query_params)
    except: return st.experimental_get_query_params()

def _qp_set(d: dict):
    try:
        st.query_params.clear()
        for k,v in d.items(): st.query_params[k] = v
    except:
        st.experimental_set_query_params(**d)

def _qp_del(k: str):
    try:
        if k in st.query_params: del st.query_params[k]
    except:
        pass

# ➜ adapta tus 3 helpers:
def _get_sid_from_url():
    q = _qp_get(); sid = q.get("sid")
    return sid if sid else None

def _set_sid_url(sid: str):
    q = _qp_get(); q["sid"] = sid; _qp_set(q)

def _clear_sid_url():
    q = _qp_get(); q.pop("sid", None); _qp_set(q)

# ==== FIN HELPERS ====


# ========= LLM CONFIG & HELPERS (opcional) ====================================
LLM_ENABLE   = os.getenv("LLM_ENABLE", "0") == "1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")      # "openai" | "ollama"
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")    # para openai
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def llm_generate_distractors(correct: str,
                             mode: str = "prompt",   # "prompt" | "concepto"
                             k: int = 3,
                             concept_hint: str = "",
                             topic: str = "") -> list[str]:
    """Devuelve hasta k distractores en español usando LLM. Si LLM está apagado o falla: []."""
    if not LLM_ENABLE:
        return []

    sys_msg = (
        "Eres un asistente para un docente de Estructuras de Datos. "
        "Devuelve opciones INCORRECTAS pero plausibles y en español. "
        "Entrega SOLO una lista de ítems, uno por línea, sin numeración, sin comillas."
    )
    if mode == "prompt":
        user_msg = (
            f"Tema: {topic or 'NA'}\n"
            f"Concepto central: {concept_hint or 'NA'}\n"
            f"Dado este prompt CORRECTO usado para generar la imagen, produce {k} prompts "
            f"alternativos que sean verosímiles pero incorrectos (mismo estilo/longitud):\n\n"
            f"CORRECTO:\n{correct}\n"
            "No repitas el significado del correcto."
        )
    else:
        user_msg = (
            f"Tema: {topic or 'Estructuras de Datos'}\n"
            f"Dado este concepto CORRECTO representado por la imagen, devuelve {k} nombres de conceptos "
            f"del mismo dominio que suelen confundirse pero sean incorrectos:\n\n"
            f"CORRECTO:\n{correct}\n"
            "Una opción por línea. No repitas el correcto."
        )

    try:
        if LLM_PROVIDER.lower() == "openai":
            try:
                from openai import OpenAI
            except Exception:
                return []
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=400,
            )
            text = (res.choices[0].message.content or "")
        elif LLM_PROVIDER.lower() == "ollama":
            import requests, json as _json
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"{sys_msg}\n\n{user_msg}",
                "stream": False,
            }
            r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate", json=payload, timeout=120)
            text = r.json().get("response", "")
        else:
            return []
    except Exception:
        return []

    # Parseo y limpieza
    out = []
    for line in text.splitlines():
        s = re.sub(r"^[\-\*\d\)\.]+\s*", "", line).strip()
        if s:
            out.append(s)

    correct_norm = _norm_txt(correct).lower()
    uniq, seen = [], set()
    for s in out:
        ss = _norm_txt(s)
        if not ss:
            continue
        if ss.lower() == correct_norm:
            continue
        if ss.lower() in seen:
            continue
        uniq.append(ss)
        seen.add(ss.lower())
        if len(uniq) >= k:
            break
    return uniq
# ========= FIN LLM CONFIG & HELPERS ===========================================

def llm_generate_mcq_items(prompt_docente: str,
                           tema: str,
                           n_items: int = 5,
                           usar_catalogo: bool = True,
                           catalog_rows: list[dict] | None = None) -> list[dict]:
    """
    Devuelve una lista de items con esquema:
      {"stem": str, "options": [4], "correct_idx": int, "image_id": str|""}
    Valida y limpia antes de retornar.
    """
    if not LLM_ENABLE:
        return []

    # 1) Prepara contexto opcional de imágenes curadas
    image_pool = []
    if usar_catalogo and catalog_rows:
        # Solo aceptadas del tema
        for r in catalog_rows:
            image_pool.append({"image_id": str(r["image_id"]), "concepto": str(r["concepto"])})

    sys_msg = (
        "Eres un generador de reactivos de opción múltiple para Estructuras de Datos. "
        "Devuelve EXCLUSIVAMENTE un JSON con una lista 'items'. Cada item tiene: "
        "stem (string), options (lista de 4 strings únicas), correct_idx (0..3), image_id (string o vacío). "
        "No incluyas explicaciones ni formato adicional."
    )

    if image_pool:
        pool_text = "\n".join([f"- {p['image_id']}: {p['concepto']}" for p in image_pool])
        user_msg = (
            f"Tema: {tema}\n"
            f"Genera {n_items} preguntas de opción múltiple en español (dificultad media), "
            f"alineadas a competencias del tema. Si puedes, asigna una imagen de esta lista "
            f"coherente con el stem (usa el image_id exacto):\n{pool_text}\n\n"
            f"Instrucción del docente: {prompt_docente}\n"
            "Responde solo con JSON válido: {\"items\": [...]}"
        )
    else:
        user_msg = (
            f"Tema: {tema}\n"
            f"Genera {n_items} preguntas de opción múltiple en español (dificultad media). "
            f"Instrucción del docente: {prompt_docente}\n"
            "Responde solo con JSON válido: {\"items\": [...]}"
        )

    # 2) Llamada al proveedor
    try:
        if LLM_PROVIDER.lower() == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":sys_msg},
                          {"role":"user","content":user_msg}],
                temperature=0.4,
                max_tokens=1200
            )
            text = (res.choices[0].message.content or "")
        else:
            import requests
            payload = {"model": OLLAMA_MODEL, "prompt": f"{sys_msg}\n\n{user_msg}", "stream": False}
            r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate", json=payload, timeout=120)
            text = r.json().get("response","")
    except Exception:
        return []

    # 3) Parseo robusto
    import re, json
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        items = data.get("items", [])
    except Exception:
        return []

    # 4) Validación y limpieza
    out = []
    for it in items:
        stem = _norm_txt(it.get("stem",""))
        opts = _ensure_unique_options(it.get("options", [])[:4])
        if len(opts) != 4 or not stem:
            continue
        try:
            cidx = int(it.get("correct_idx", 0))
        except Exception:
            cidx = 0
        cidx = max(0, min(3, cidx))
        img_id = _norm_txt(it.get("image_id",""))
        if image_pool and img_id and img_id not in [p["image_id"] for p in image_pool]:
            img_id = ""  # si la IA inventa uno, lo vaciamos
        out.append({"stem": stem, "options": opts, "correct_idx": cidx, "image_id": img_id})
    return out


# ==== GENERADOR AUTOMÁTICO DE PREGUNTAS DESDE CATÁLOGO (Bloque B) ====
def auto_generate_questions_from_catalog(
    df_meta: pd.DataFrame,
    mode: str = "concepto",         # "concepto" | "prompt"
    topic_filter: str | None = None,
    state_filter: str = "aceptar",   # "aceptar" | "pendiente" | "ajustar" | "descartar" | "(todos)"
    n_questions: int = 5,
    k_distractors: int = 3,
    strategy: str = "mix",           # "tfidf" | "mutacion" | "llm" | "mix"
    seed: int | None = None
) -> list[dict]:
    """
    Devuelve una lista de items tipo:
        {"stem": str, "options": [str,str,str,str], "correct_idx": int,
         "image_path": str, "image_id": str}
    mode="concepto": opciones son conceptos; stem: '¿Qué concepto representa la imagen?'
    mode="prompt"  : opciones son prompts;  stem: 'Selecciona el prompt correcto...'
    """
    rng = random.Random(seed or 0)

    # --- 1) Filtrar base ---
    view = df_meta.copy()
    # imagen existente en disco
    view["ruta"] = view["ruta"].astype(str).fillna("")
    view = view[view["ruta"].apply(lambda p: bool(_safe_path(p)))]

    if topic_filter and topic_filter not in ("", "(todos)"):
        view = view[view["tema"] == topic_filter]

    if state_filter and state_filter not in ("", "(todos)"):
        view = view[view["estado"] == state_filter]

    if mode == "prompt":
        # solo filas con prompt no vacío
        view = view[view["prompt"].fillna("").astype(str).str.strip() != ""]

    if view.empty:
        return []

    # --- 2) Preparar corpus para distractores ---
    # corpus de prompts (para TF-IDF o LLM en modo prompt)
    corpus_prompts = view["prompt"].fillna("").astype(str).tolist()
    # pool de conceptos (para modo concepto)
    pool_conceptos = view["concepto"].fillna("").astype(str).tolist()

    # --- 3) Sampling de imágenes ---
    # Tomamos un subconjunto de filas al azar (sin replacement) hasta n_questions
    idxs = list(view.index)
    rng.shuffle(idxs)
    idxs = idxs[:min(n_questions, len(idxs))]

    out_items = []

    for idx in idxs:
        row = view.loc[idx]
        img_path = _safe_path(row.get("ruta", ""))
        img_id   = _safe_path(row.get("image_id", ""))
        tema     = _norm_txt(row.get("tema", ""))
        concepto = _norm_txt(row.get("concepto", ""))
        prompt   = _norm_txt(row.get("prompt", ""))

        if mode == "concepto":
            correct = concepto if concepto else "concepto"
            # Distractores por similitud de texto en el pool + LLM opcional
            dists = _pick_concept_distractors(correct, k=k_distractors)
            if LLM_ENABLE and strategy in ("llm", "mix"):
                d_llm = llm_generate_distractors(correct, mode="concepto", k=k_distractors, concept_hint=concepto, topic=tema)
                dists += d_llm
            # fallback si falta: usa conceptos aleatorios del pool
            if len(dists) < k_distractors:
                fallback = [c for c in pool_conceptos if _norm_txt(c).lower() != _norm_txt(correct).lower()]
                rng.shuffle(fallback)
                dists += fallback[: (k_distractors - len(dists))]

            opts = _ensure_unique_options([correct] + dists)[: (1 + k_distractors)]
            # barajar y calcular índice correcto
            rng.shuffle(opts)
            correct_idx = opts.index(correct) if correct in opts else 0

            stem = f"¿Qué concepto representa la imagen mostrada? (Tema: {tema})"

        else:  # mode == "prompt"
            # Si no hay prompt, salta esta fila
            if not prompt:
                continue

            # piscina de distractores
            cand = []

            if strategy in ("tfidf", "mix"):
                # similares por TF-IDF
                cand += _tfidf_distractors(prompt, [p for p in corpus_prompts if _norm_txt(p)], k=k_distractors*2)

            if strategy in ("mutacion", "mix"):
                # varias mutaciones del prompt correcto
                mset = set()
                for _ in range(k_distractors*2):
                    m = _mutate_prompt(prompt)
                    m = _norm_txt(m)
                    if m and m.lower() != _norm_txt(prompt).lower():
                        mset.add(m)
                    if len(mset) >= k_distractors*2:
                        break
                cand += list(mset)

            if LLM_ENABLE and strategy in ("llm", "mix"):
                cand += llm_generate_distractors(prompt, mode="prompt", k=k_distractors, concept_hint=concepto, topic=tema)

            # limpieza + recorte
            cand = [c for c in _ensure_unique_options(cand) if _norm_txt(c).lower() != _norm_txt(prompt).lower()]
            if len(cand) < k_distractors:
                # fallback: mini variaciones
                while len(cand) < k_distractors:
                    cand.append(_mutate_prompt(prompt))
                cand = _ensure_unique_options(cand)

            opts = _ensure_unique_options([prompt] + cand)[: (1 + k_distractors)]
            rng.shuffle(opts)
            correct_idx = opts.index(prompt) if prompt in opts else 0

            stem = f"Selecciona el prompt correcto usado para generar esta imagen (Tema: {tema}, Concepto: {concepto})."

        item = {
            "stem": stem,
            "options": opts,
            "correct_idx": int(correct_idx),
            "image_path": img_path,
            "image_id": img_id
        }
        out_items.append(item)

    return out_items
# ==== FIN Bloque B ====


# Asegura carpetas
BASE = Path("data")
IMG_DIR = BASE / "imagenes"
BASE.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# ===== Config simple (PIN docente y código de clase) =====
CONFIG_PATH = BASE / "config.json"

DEFAULT_CONFIG = {
    "DOCENTE_PIN": "12345",        
    "CLASS_CODE": "UTA-2026"   
}

def slugify(s: str, maxlen: int = 24) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^A-Za-z0-9-_]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s[:maxlen] if len(s) > maxlen else s


def _img_b64(path: Path):
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None

def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        return DEFAULT_CONFIG
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
CFG = load_config()

# --- Session Manager (persistencia de login) ---
SESSIONS_PATH = BASE / "sessions.json"
def _sess_load():
    return (json.load(open(SESSIONS_PATH, "r", encoding="utf-8"))
            if SESSIONS_PATH.exists() else {})

def _sess_save(d):
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# --- Query params helpers (SOLO API nueva) ---
def _get_sid_from_url():
    # Devuelve None si no hay ?sid=
    sid = st.query_params.get("sid")
    return sid if sid else None

def _set_sid_url(sid: str):
    st.query_params["sid"] = sid

def _clear_sid_url():
    if "sid" in st.query_params:
        del st.query_params["sid"]
    else:
        st.query_params.clear()


def restore_auth_from_sid():
    """Restaura auth si hay ?sid= en la URL y no estamos logueados."""
    sid = _get_sid_from_url()
    if sid and not st.session_state.get("auth_ok", False):
        sessions = _sess_load()
        info = sessions.get(sid)
        if info and info.get("auth_ok"):
            st.session_state.update({
                "auth_ok": True,
                "auth_role": info.get("auth_role"),
                "auth_name": info.get("auth_name"),
                "class_code": info.get("class_code", ""),
                "sid": sid
            })
            return True
    return False

def persist_current_auth():
    """Guarda la sesión actual en disco y deja ?sid= en la URL."""
    sid = st.session_state.get("sid") or uuid4().hex
    st.session_state["sid"] = sid
    sessions = _sess_load()
    sessions[sid] = {
        "auth_ok": True,
        "auth_role": st.session_state.get("auth_role"),
        "auth_name": st.session_state.get("auth_name"),
        "class_code": st.session_state.get("class_code", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds")
    }
    _sess_save(sessions)
    _set_sid_url(sid)

def clear_persisted_auth():
    """Cierra sesión: borra de disco y limpia la URL."""
    sid = st.session_state.get("sid")
    sessions = _sess_load()
    if sid in sessions:
        del sessions[sid]
        _sess_save(sessions)
    for k in ["auth_ok", "auth_role", "auth_name", "class_code", "sid"]:
        st.session_state.pop(k, None)
    _clear_sid_url()

# ===== Estado de autenticación =====
# Estado base
for k, v in {"auth_ok": False, "auth_role": None, "auth_name": "", "class_code": "", "last_login_err": ""}.items():
    if k not in st.session_state: st.session_state[k] = v

# Restaura si hay ?sid= y no estamos logueados aún
restore_auth_from_sid()

# -------------------------------------------------------
# LOGIN MEJORADO V3 - GEN-EDViz Monitor
# Diseño más impactante con mejores colores y animaciones
# REEMPLAZA desde "if not st.session_state.auth_ok:" hasta "st.stop()"
# -------------------------------------------------------

if not st.session_state.auth_ok:
    # ========== CSS COMPLETO ==========
    st.markdown("""
    <style>
    /* ===== IMPORTS DE FUENTES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* ===== RESET Y OCULTAR ELEMENTOS STREAMLIT ===== */
    [data-testid='stSidebar'],
    [data-testid='stHeader'],
    header, footer, #MainMenu,
    .stDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    
    section.main > div.block-container {
        padding: 1rem 2rem !important;
        max-width: 100% !important;
    }
    
    .stApp {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* ===== FONDO CON GRADIENTE ANIMADO ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 75%, #0a0a1a 100%) !important;
        background-size: 400% 400% !important;
        animation: gradientFlow 20s ease infinite !important;
        min-height: 100vh !important;
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        25% { background-position: 50% 100%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 0%; }
    }
    
    /* ===== EFECTO DE ESTRELLAS/PARTÍCULAS ===== */
    .stars-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        animation: twinkle 3s ease-in-out infinite;
    }
    
    .star:nth-child(1) { width: 2px; height: 2px; top: 10%; left: 20%; animation-delay: 0s; }
    .star:nth-child(2) { width: 3px; height: 3px; top: 20%; left: 80%; animation-delay: 0.5s; }
    .star:nth-child(3) { width: 2px; height: 2px; top: 40%; left: 10%; animation-delay: 1s; }
    .star:nth-child(4) { width: 4px; height: 4px; top: 15%; left: 50%; animation-delay: 1.5s; }
    .star:nth-child(5) { width: 2px; height: 2px; top: 70%; left: 85%; animation-delay: 2s; }
    .star:nth-child(6) { width: 3px; height: 3px; top: 80%; left: 30%; animation-delay: 0.3s; }
    .star:nth-child(7) { width: 2px; height: 2px; top: 50%; left: 70%; animation-delay: 0.8s; }
    .star:nth-child(8) { width: 3px; height: 3px; top: 30%; left: 40%; animation-delay: 1.2s; }
    .star:nth-child(9) { width: 2px; height: 2px; top: 60%; left: 5%; animation-delay: 1.8s; }
    .star:nth-child(10) { width: 4px; height: 4px; top: 85%; left: 60%; animation-delay: 2.2s; }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    /* ===== ORBE DECORATIVO ===== */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        pointer-events: none;
        z-index: 0;
    }
    
    .orb-1 {
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%);
        top: -100px;
        left: -100px;
        animation: float1 15s ease-in-out infinite;
    }
    
    .orb-2 {
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.25) 0%, transparent 70%);
        bottom: -50px;
        right: -50px;
        animation: float2 18s ease-in-out infinite;
    }
    
    .orb-3 {
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 70%);
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: float3 12s ease-in-out infinite;
    }
    
    @keyframes float1 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, 30px); }
    }
    
    @keyframes float2 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(-20px, -20px); }
    }
    
    @keyframes float3 {
        0%, 100% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.1); }
    }
    
    /* ===== LOGO CONTAINER ===== */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 12px;
    }
    
    .logo-icon {
        font-size: 4rem;
        animation: bounce 2s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* ===== TÍTULO PRINCIPAL ===== */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 30%, #f472b6 50%, #818cf8 70%, #a78bfa 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        margin: 0;
        letter-spacing: -2px;
        line-height: 1;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    
    /* ===== DESCRIPCIÓN ===== */
    .description-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 24px 0;
        backdrop-filter: blur(10px);
    }
    
    .description-text {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0;
    }
    
    .highlight {
        color: #c4b5fd;
        font-weight: 600;
    }
    
    /* ===== FEATURE CARDS ===== */
    .features-grid {
        display: flex;
        gap: 16px;
        margin-top: 20px;
    }
    
    .feature-card {
        flex: 1;
        background: linear-gradient(145deg, rgba(30, 30, 50, 0.8) 0%, rgba(20, 20, 35, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
        background-size: 200% auto;
        animation: shimmer 2s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.2);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
        display: block;
    }
    
    .feature-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.4;
    }
    
    /* ===== LOGIN CARD ===== */
    .login-card {
        background: linear-gradient(165deg, rgba(25, 25, 40, 0.95) 0%, rgba(15, 15, 25, 0.98) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 24px;
        padding: 32px;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 25px 60px rgba(0, 0, 0, 0.5),
            0 0 100px rgba(139, 92, 246, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .login-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #8b5cf6, #ec4899, #3b82f6, #8b5cf6);
        background-size: 300% auto;
        animation: shimmer 4s linear infinite;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 24px;
    }
    
    .login-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    
    /* ===== TABS MEJORADOS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(0, 0, 0, 0.3);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748b;
        background: transparent;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #a78bfa;
        background: rgba(139, 92, 246, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* ===== ROLE INFO BOX ===== */
    .role-info {
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .role-info.docente {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-left: 4px solid #8b5cf6;
    }
    
    .role-info.estudiante {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 4px solid #10b981;
    }
    
    .role-icon {
        font-size: 1.8rem;
    }
    
    .role-text strong {
        display: block;
        font-size: 1rem;
        margin-bottom: 2px;
    }
    
    .role-text span {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    /* ===== INPUTS MEJORADOS ===== */
    .stTextInput > label { display: none !important; }
    
    .stTextInput > div > div > input {
        font-family: 'Outfit', sans-serif !important;
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #4a5568 !important;
    }
    
    /* ===== BOTÓN PRINCIPAL ===== */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 32px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5) !important;
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 50%, #7c3aed 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
    }
    
    .footer p {
        color: #475569;
        font-size: 0.85rem;
        margin: 0;
    }
    
    .footer strong {
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-title { font-size: 2.5rem; }
        .features-grid { flex-direction: column; }
        .login-card { padding: 24px; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ===== ELEMENTOS DECORATIVOS =====
    st.markdown("""
    <div class="stars-container">
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div>
    </div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    """, unsafe_allow_html=True)
    
    # ===== LAYOUT PRINCIPAL =====
    col_info, col_login = st.columns([1.15, 0.85], gap="large")
    
    # ===== COLUMNA IZQUIERDA - INFORMACIÓN =====
    with col_info:
        # Logo y título
        st.markdown("""
        <div class="logo-container">
            <span class="logo-icon">🧠</span>
        </div>
        <h1 class="main-title">GEN-EDViz</h1>
        <p class="subtitle">Monitor de Imágenes Educativas con IA</p>
        """, unsafe_allow_html=True)
        
        # Descripción
        st.markdown("""
        <div class="description-card">
            <p class="description-text">
                Transforma la enseñanza de Estructuras de Datos con el poder de la Inteligencia Artificial. 
Genera,         evalúa y optimiza recursos visuales que potencian el aprendizaje de tus estudiantes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <span class="feature-icon">🎨</span>
                <div class="feature-title">Generación IA</div>
                <div class="feature-desc">DALL-E 3 y GPT-4o para crear contenido educativo</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Evaluación de Calidad</div>
                <div class="feature-desc">Evaluación de gráfico y prompt generado por IA</div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🧪</span>
                <div class="feature-title">Rúbrica 4x4</div>
                <div class="feature-desc">Mide el impacto en el aprendizaje estudiantil</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== COLUMNA DERECHA - LOGIN =====
    # ===== COLUMNA DERECHA - LOGIN =====
    with col_login:
        # Logo del proyecto        
        # Logo dentro del card de login
        logo_path = Path("assets/logo_genedviz.png")
        if logo_path.exists():
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(f"""
            <div class="login-card">
                <div class="login-header">
                    <img src="data:image/png;base64,{logo_b64}" 
                        style="width: 250px; height: 200px; border-radius: 16px; 
                                margin-bottom: 16px; display: block; margin-left: auto; margin-right: auto;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="login-card">
                <div class="login-header">
                    <h2 class="login-title">✨ Iniciar Sesión</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs
        tab_doc, tab_est = st.tabs(["🎓 Docente", "📚 Estudiante"])
        
        # ===== TAB DOCENTE =====
        with tab_doc:
            st.markdown("""
            <div class="role-info docente">
                <span class="role-icon">🎓</span>
                <div class="role-text">
                    <strong style="color: #c4b5fd;">Acceso Docente</strong>
                    <span style="color: #a78bfa;">Gestión completa del sistema</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            name_d = st.text_input("Nombre", placeholder="👤 Tu nombre completo", key="v3_doc_name")
            code_d = st.text_input("Código", placeholder="🏫 Código de clase (Ej: UTA-2026)", 
                                   key="v3_doc_code", value=st.session_state.get("class_code", ""))
            pin_d = st.text_input("PIN", placeholder="🔑 PIN Docente", type="password", key="v3_doc_pin")
            
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            
            if st.button("🚀 Iniciar Sesión", key="v3_btn_doc"):
                if not name_d.strip():
                    st.error("⚠️ Ingresa tu nombre")
                elif pin_d.strip() != CFG.get("DOCENTE_PIN", ""):
                    st.error("❌ PIN incorrecto")
                else:
                    st.session_state.auth_ok = True
                    st.session_state.auth_role = "Docente"
                    st.session_state.auth_name = name_d.strip()
                    st.session_state.class_code = code_d.strip()
                    persist_current_auth()
                    st.success("✅ ¡Bienvenido!")
                    time.sleep(0.5)
                    st.rerun()
        
        # ===== TAB ESTUDIANTE =====
        with tab_est:
            st.markdown("""
            <div class="role-info estudiante">
                <span class="role-icon">📚</span>
                <div class="role-text">
                    <strong style="color: #6ee7b7;">Acceso Estudiante</strong>
                    <span style="color: #34d399;">Vizualización de la Galería y Participación Rúbrica 4x4</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            name_e = st.text_input("Nombre", placeholder="👤 Tu nombre completo", key="v3_est_name")
            code_e = st.text_input("Código", placeholder="🏫 Código de clase (Ej: UTA-2026)", 
                                   key="v3_est_code", value=st.session_state.get("class_code", ""))
            
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            
            if st.button("🎯 Unirse a la Clase", key="v3_btn_est"):
                if not name_e.strip():
                    st.error("⚠️ Ingresa tu nombre")
                elif code_e.strip() != CFG.get("CLASS_CODE", ""):
                    st.error("❌ Código de clase inválido")
                else:
                    st.session_state.auth_ok = True
                    st.session_state.auth_role = "Estudiante"
                    st.session_state.auth_name = name_e.strip()
                    st.session_state.class_code = code_e.strip()
                    persist_current_auth()
                    st.success("✅ ¡Bienvenido!")
                    time.sleep(0.5)
                    st.rerun()
    
    # ===== DESACTIVAR AUTOCOMPLETADO =====
    st.markdown("""
    <script>
        setTimeout(function() {
            const inputs = document.querySelectorAll('input[type="text"], input[type="password"]');
            inputs.forEach(function(input) {
                input.setAttribute('autocomplete', 'off');
                input.setAttribute('autocorrect', 'off');
                input.setAttribute('autocapitalize', 'off');
                input.setAttribute('spellcheck', 'false');
            });
        }, 500);
    </script>
    
    <style>
        input:-webkit-autofill {
            -webkit-box-shadow: 0 0 0 30px rgba(0, 0, 0, 0.4) inset !important;
            -webkit-text-fill-color: #f1f5f9 !important;
        }
        
        input::-webkit-contacts-auto-fill-button,
        input::-webkit-credentials-auto-fill-button {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown("""
    <div class="footer">
        <p><strong>Universidad Técnica de Ambato</strong> · 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()       
    
    # ===== FOOTER =====
    st.markdown("""
    <div class="footer">
        <p><strong>Universidad Técnica de Ambato</strong> · 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# ========== ESTILOS GLOBALES PARA TODA LA APP ==========
st.markdown("""
<style>
/* ===== IMPORTAR FUENTE ===== */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ===== FONDO GENERAL ===== */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0f0f23 100%) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0 !important;
}

/* ===== TÍTULOS ===== */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
}

h1 {
    background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
}

/* ===== TEXTO GENERAL ===== */
p, span, label, .stMarkdown {
    color: #cbd5e1 !important;
}

/* ===== INPUTS ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: rgba(15, 15, 30, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Outfit', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
}

/* ===== SELECTBOX ===== */
.stSelectbox > div > div {
    background: rgba(15, 15, 30, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
}

/* ===== BOTONES ===== */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(139, 92, 246, 0.1) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    color: #a78bfa !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 15, 30, 0.6) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 6px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    color: white !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: rgba(139, 92, 246, 0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ===== DATAFRAMES ===== */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ===== MÉTRICAS ===== */
[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(139, 92, 246, 0.05) !important;
    border: 2px dashed rgba(139, 92, 246, 0.3) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    background: rgba(139, 92, 246, 0.1) !important;
}

/* ===== SLIDER ===== */
.stSlider > div > div > div > div {
    background: #8b5cf6 !important;
}

/* ===== RADIO BUTTONS ===== */
.stRadio > div {
    background: rgba(15, 15, 30, 0.4) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ===== CHECKBOX ===== */
.stCheckbox > label > span {
    color: #e2e8f0 !important;
}

/* ===== SUCCESS/ERROR/WARNING/INFO ===== */
.stSuccess {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    border-radius: 12px !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
}

.stWarning {
    background: rgba(251, 191, 36, 0.1) !important;
    border: 1px solid rgba(251, 191, 36, 0.3) !important;
    border-radius: 12px !important;
}

.stInfo {
    background: rgba(59, 130, 246, 0.1) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 12px !important;
}

/* ===== DIVIDER ===== */
hr {
    border-color: rgba(139, 92, 246, 0.2) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0f0f1a;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #a78bfa, #8b5cf6);
}

/* ===== SIDEBAR RADIO (MENU) ===== */
[data-testid="stSidebar"] .stRadio > div {
    background: transparent !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    padding: 10px 16px !important;
    border-radius: 10px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(139, 92, 246, 0.15) !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(124, 58, 237, 0.2) 100%) !important;
    border-left: 3px solid #8b5cf6 !important;
}

/* ===== HEADER DE SECCIONES ===== */
.section-header {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
}

/* ===== CARDS ===== */
.custom-card {
    background: linear-gradient(145deg, rgba(25, 25, 40, 0.9) 0%, rgba(15, 15, 25, 0.95) 100%);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

.custom-card:hover {
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.1);
}
</style>
""", unsafe_allow_html=True)

is_docente = (st.session_state.auth_role == "Docente")
user_name = st.session_state.auth_name

DOCENTE_SECTIONS = [
    # FASE 1: Preparación y Generación
    "Generador IA",
    # FASE 2: Evaluación
    "Evaluación IA (imagen + prompt)",
    "Evaluar (rúbrica 4x4)",
    # FASE 3: Visualización
    "Galería",
    "Resumen y reportes"
]

ESTUDIANTE_SECTIONS = [
    "Galería", "Evaluar (rúbrica 4x4)"
]


st.sidebar.title("GEN-EDViz Monitor")
st.sidebar.caption(f"{user_name} · {st.session_state.auth_role}")
secciones = DOCENTE_SECTIONS if is_docente else ESTUDIANTE_SECTIONS

# ========== PERSISTENCIA DE SECCIÓN EN URL ==========
def _get_seccion_from_url():
    """Obtiene la sección guardada en la URL."""
    try:
        q = dict(st.query_params)
        sec = q.get("seccion", "")
        # Decodificar espacios y caracteres especiales
        if sec:
            from urllib.parse import unquote
            sec = unquote(sec)
        return sec if sec in secciones else None
    except:
        return None

def _set_seccion_url(seccion_name: str):
    """Guarda la sección en la URL."""
    try:
        from urllib.parse import quote
        # Mantener otros parámetros (como sid)
        current = dict(st.query_params)
        current["seccion"] = quote(seccion_name)
        st.query_params.clear()
        for k, v in current.items():
            st.query_params[k] = v
    except:
        pass

# Obtener sección de la URL o usar la primera por defecto
seccion_guardada = _get_seccion_from_url()

# Determinar el índice inicial
if seccion_guardada and seccion_guardada in secciones:
    indice_inicial = secciones.index(seccion_guardada)
else:
    indice_inicial = 0

# Radio button con el índice correcto
seccion = st.sidebar.radio(
    "Secciones", 
    secciones, 
    index=indice_inicial,
    key="nav_seccion"
)

# Guardar la sección actual en la URL (solo si cambió)
if "ultima_seccion" not in st.session_state:
    st.session_state.ultima_seccion = seccion

if seccion != st.session_state.ultima_seccion:
    st.session_state.ultima_seccion = seccion
    _set_seccion_url(seccion)

# También guardar en URL si es la primera vez
if not seccion_guardada:
    _set_seccion_url(seccion)

# Cerrar sesión (opcional)
if st.sidebar.button("Cerrar sesión"):
    clear_persisted_auth()
    (st.rerun if hasattr(st,"rerun") else st.experimental_rerun)()


# Rutas UNIFICADAS con Path (no volver a redefinir IMG_DIR como str)
DATA_DIR = BASE                      # Path("data")
IMG_DIR  = IMG_DIR                   # ya es BASE / "imagenes"
CSV_PATH = DATA_DIR / "evaluaciones.csv"
META_PATH = DATA_DIR / "catalogo.csv"
PERCEP_PATH = DATA_DIR / "percepcion.csv"

# Asegura directorios
DATA_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# --- LOG DE EVENTOS (events.csv) ---
EVENTS = DATA_DIR / "events.csv"
if not EVENTS.exists():
    EVENTS.write_text("ts,user,role,action,ref_id,extra_json\n", encoding="utf-8")

def log_event(action, ref_id: str = "", extra: dict | None = None):
    # convierte extra a JSON seguro (por si hay Paths o tipos numpy)
    def _jsonable(v):
        import numpy as np
        from pathlib import Path
        if isinstance(v, Path): return v.as_posix()
        if isinstance(v, (np.integer, np.floating)): return v.item()
        try:
            json.dumps(v, ensure_ascii=False); return v
        except Exception:
            return str(v)
    extra = {k: _jsonable(v) for k, v in (extra or {}).items()}
    row = ",".join([
        datetime.now().isoformat(timespec="seconds"),
        st.session_state.get("auth_name",""),
        st.session_state.get("auth_role",""),
        action,
        ref_id,
        json.dumps(extra, ensure_ascii=False)
    ]) + "\n"
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(row)


# Directorio para retroalimentación colaborativa
FEED_DIR = Path("data") / "feedback"
FEED_DIR.mkdir(parents=True, exist_ok=True)

# Directorio para quizzes (futuro)
QUIZ_DIR = Path("data") / "quizzes"
QUIZ_DIR.mkdir(parents=True, exist_ok=True)

def jload(p: Path, default=None):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def jsave(p: Path, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# === QUIZ: estado por usuario (autodirigido) ===
def _user_state_path(qdir: Path) -> Path:
    return qdir / "user_state.json"

def _load_user_state(qdir: Path) -> dict:
    stobj = jload(_user_state_path(qdir), default=None)
    if not stobj:
        stobj = {"users": {}}  # user -> {idx, finished, started_at, finished_at}
    return stobj

def _save_user_state(qdir: Path, stobj: dict):
    jsave(_user_state_path(qdir), stobj)

def _get_or_init_user_state(qdir: Path, user: str) -> dict:
    stobj = _load_user_state(qdir)
    if user not in stobj["users"]:
        stobj["users"][user] = {
            "idx": 0,
            "finished": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None
        }
        _save_user_state(qdir, stobj)
    return stobj["users"][user]

def _update_user_state(qdir: Path, user: str, **patch):
    stobj = _load_user_state(qdir)
    if user not in stobj["users"]:
        stobj["users"][user] = {
            "idx": 0,
            "finished": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None
        }
    stobj["users"][user].update(patch)
    _save_user_state(qdir, stobj)

def _upsert_answer(ans_path: Path, quiz_id: str, user: str, q_idx: int, choice_idx: int, correct: int):
    cols = ["quiz_id","user","q_idx","choice_idx","correct","timestamp"]
    df = pd.read_csv(ans_path) if ans_path.exists() and ans_path.stat().st_size > 0 else pd.DataFrame(columns=cols)
    mask = (df["quiz_id"] == quiz_id) & (df["user"] == user) & (df["q_idx"] == q_idx)
    if mask.any():
        df.loc[mask, ["choice_idx","correct","timestamp"]] = [choice_idx, correct, datetime.now().isoformat(timespec="seconds")]
    else:
        df = pd.concat([df, pd.DataFrame([{
            "quiz_id": quiz_id, "user": user, "q_idx": q_idx,
            "choice_idx": choice_idx, "correct": correct,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }])], ignore_index=True)
    df.to_csv(ans_path, index=False)
    return df

def _recompute_user_score(qdir: Path, quiz_id: str, user: str) -> int:
    ans_path = qdir / "answers.csv"
    score = 0
    if ans_path.exists() and ans_path.stat().st_size > 0:
        tmp = pd.read_csv(ans_path)
        score = int(tmp[(tmp["quiz_id"] == quiz_id) & (tmp["user"] == user)]["correct"].fillna(0).astype(int).sum())
    roster_path = qdir / "roster.csv"
    roster_df = pd.read_csv(roster_path) if roster_path.exists() and roster_path.stat().st_size > 0 else pd.DataFrame(columns=["user","joined_at","score"])
    if user in roster_df["user"].tolist():
        roster_df.loc[roster_df["user"] == user, "score"] = score
    else:
        roster_df = pd.concat([roster_df, pd.DataFrame([{
            "user": user, "joined_at": datetime.now().isoformat(timespec="seconds"), "score": score
        }])], ignore_index=True)
    roster_df.to_csv(roster_path, index=False)
    return score

# ==== HELPERS PARA QUIZZES (para análisis pre/post) ====

def list_quizzes():
    """
    Devuelve una lista de quizzes encontrados en QUIZ_DIR, cada uno como:
    {"quiz_id", "title", "quiz_role", "path"}
    """
    quizzes = []
    if QUIZ_DIR.exists():
        for sub in QUIZ_DIR.iterdir():
            if sub.is_dir():
                q = jload(sub / "quiz.json", default=None)
                if q:
                    quizzes.append({
                        "quiz_id": q.get("quiz_id", sub.name),
                        "title": q.get("title", q.get("quiz_id", sub.name)),
                        "quiz_role": q.get("quiz_role", "other"),
                        "path": sub
                    })
    return quizzes

def load_roster_for_quiz(qinfo: dict) -> pd.DataFrame:
    """
    Carga el roster (usuario + score) para un quiz.
    """
    qdir = qinfo["path"]
    roster_path = qdir / "roster.csv"
    if roster_path.exists() and roster_path.stat().st_size > 0:
        df = pd.read_csv(roster_path)
        if "user" not in df.columns:
            df["user"] = ""
        if "score" not in df.columns:
            df["score"] = 0
        df["user"] = df["user"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        return df[["user","score"]]
    else:
        # si no hay roster, devolvemos vacío
        return pd.DataFrame(columns=["user","score"])


# -------------------------------------------------------
# Utilidades
# -------------------------------------------------------
CRITERIA = [
    ("fidelidad", "¿Representa correctamente el concepto (estructuras, relaciones, pasos)?"),
    ("claridad", "¿Se lee y entiende bien (legibilidad, señalización, foco visual)?"),
    ("pertinencia", "¿Está alineada al objetivo/competencia del tema del curso?"),
    ("equidad", "¿Cumple criterios de accesibilidad y neutralidad cultural/estética?")
]

# Descriptores de nivel 1–4 por criterio (para mostrar al evaluador)
LEVEL_DESC = {
    "fidelidad": {
        1: "Errores graves de concepto (estructura mal representada o paso incorrecto).",
        2: "Errores/modificaciones moderadas; requiere muchas aclaraciones del docente.",
        3: "Correcta en lo esencial, con detalles mejorables.",
        4: "Representación precisa y coherente con el concepto."
    },
    "claridad": {
        1: "Muy confusa o recargada; difícil de leer.",
        2: "Elementos legibles pero con desorden o ruido visual.",
        3: "Generalmente clara, con algunos elementos mejorables.",
        4: "Muy clara, jerarquía visual evidente y foco bien definido."
    },
    "pertinencia": {
        1: "No se ajusta al objetivo del tema o induce a malentendidos.",
        2: "Parcialmente alineada; mezcla información irrelevante.",
        3: "Pertinente al objetivo, con detalles adicionales no críticos.",
        4: "Directamente alineada al objetivo y al resultado de aprendizaje."
    },
    "equidad": {
        1: "Contiene sesgos evidentes o problemas serios de accesibilidad.",
        2: "Muestra posibles sesgos o baja accesibilidad (contrastes, texto, etc.).",
        3: "En general neutra y relativamente accesible, con mejoras posibles.",
        4: "Respetuosa, diversa y con buena accesibilidad visual."
    }
}

# Opciones para la decisión global de uso
DECISION_USO = {
    "usar_sin_cambios": "Sí, la usaría tal cual en clase.",
    "usar_con_ajustes": "La usaría con ajustes o explicaciones adicionales.",
    "no_usar_en_clase": "No la usaría en un contexto de enseñanza real."
}

SEVERIDAD_GLOBAL = {
    "ninguno": "Sin problemas relevantes.",
    "menores": "Problemas menores, poco visibles.",
    "moderados": "Problemas moderados, visibles pero manejables.",
    "graves": "Problemas graves que afectan su uso pedagógico."
}

def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """
    Calcula alfa de Cronbach a partir de un DataFrame de ítems Likert
    (filas = personas, columnas = ítems numéricos).
    """
    if items_df is None or items_df.empty:
        return np.nan

    # Eliminar filas totalmente vacías
    items = items_df.dropna(how="all")
    if items.empty or items.shape[1] < 2:
        return np.nan

    k = items.shape[1]  # número de ítems
    var_items = items.var(axis=0, ddof=1)
    var_total = items.sum(axis=1).var(ddof=1)

    if var_total <= 0:
        return np.nan

    return (k / (k - 1.0)) * (1.0 - var_items.sum() / var_total)


def load_csv(path, cols=None):
    path = str(path)  # por si viene como Path
    if os.path.exists(path):
        df = pd.read_csv(path)
        if cols:
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
        return df
    return pd.DataFrame(columns=cols or [])

def save_csv(df, path):
    df.to_csv(str(path), index=False)


# -------------------------------------------------------
# Datos
# -------------------------------------------------------
evaluaciones_cols = [
    "image_id","tema","concepto","rater","criterio","puntaje","comentario",
    "decision_uso","severidad","comentario_global","timestamp"
]
df_eval = load_csv(CSV_PATH, evaluaciones_cols)


catalogo_cols = [
    "image_id","tema","concepto","herramienta","prompt","version","fecha",
    "autor","alt_text","ruta","wcag_ratio","edge_density","estado"
]
df_meta = load_csv(META_PATH, catalogo_cols)

percepcion_cols = [
    "user","role","tema",
    "claridad","utilidad","aprendizaje","motivacion","sesgos",
    "comentario","timestamp"
]
df_percep = load_csv(PERCEP_PATH, percepcion_cols)


# -------------------------------------------------------
# Subir y catalogar
# -------------------------------------------------------
if seccion == "Subir y catalogar":
    if not is_docente:
        st.warning("Solo el Docente puede subir y catalogar imágenes.")
        st.stop()
    st.header("Subir y catalogar imágenes")
    tema = st.text_input("Tema (p. ej., arboles, grafos, pilas)", "")
    concepto = st.text_input("Concepto (p. ej., inorden, BFS, push/pop)", "")
    herramienta = st.selectbox("Herramienta", ["Gemini", "DALL·E", "Stable Diffusion", "Midjourney"])
    prompt = st.text_area("Prompt usado")
    alt_text = st.text_area("Texto alternativo (accesibilidad)")

    archivo = st.file_uploader("Cargar imagen (.png/.jpg)", type=["png","jpg","jpeg"])
    autor = st.text_input("Autor/Equipo", "")
    version = st.text_input("Versión", "v1")

    # --- Guardar en catálogo (todo dentro del botón) ---
    if st.button(
        "Guardar en catálogo",
        type="primary",
        disabled=(archivo is None or tema.strip()=="" or concepto.strip()=="")
    ):
        # 1) Bytes e ID
        img_bytes = archivo.read()
        image_id = f"IMG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # 2) Nombre corto y seguro
        tema_s = slugify(tema)
        herr_s = slugify(herramienta)
        hoy_s  = date.today().isoformat()
        uid8   = str(uuid4())[:8]
        filename = f"IMG-{hoy_s}-{tema_s}-{herr_s}-{uid8}.png"
        ruta = IMG_DIR / filename  # Path

        # 3) Guardar binario
        with open(ruta, "wb") as f:
            f.write(img_bytes)

        # 4) Métricas objetivas
        img = Image.open(io.BytesIO(img_bytes))
        ratio = wcag_like_contrast_ratio(img)
        edens = edge_density(img)

        # 5) Metadatos
        row = {
            "image_id": image_id,
            "tema": tema,
            "concepto": concepto,
            "herramienta": herramienta,
            "prompt": prompt,
            "version": version,
            "fecha": str(datetime.now().date()),
            "autor": user_name,
            "alt_text": alt_text,
            "ruta": str(ruta.as_posix()),
            "wcag_ratio": round(ratio, 2),
            "edge_density": round(edens, 3),
            "estado": "pendiente"
        }

        # 6) Persistir
        df_meta = pd.concat([df_meta, pd.DataFrame([row])], ignore_index=True)
        save_csv(df_meta, META_PATH)

        st.success(
            f"Guardado {image_id} ({filename}). "
            f"Contraste≈{row['wcag_ratio']} | Edge dens={row['edge_density']}."
        )
        log_event("catalog.save", ref_id=image_id, extra=row)

    st.divider()

# -------------------------------------------------------
# 🎨 GENERADOR DE IMÁGENES IA (GPT-IMAGE 1.5)
# -------------------------------------------------------
elif seccion == "Generador IA":
    st.header("🎨 Generador de Imágenes con IA")
    st.caption("Genera imágenes educativas de estructuras de datos usando GPT-Image 1.5")
    
    if not is_docente:
        st.warning("Solo el Docente puede generar imágenes.")
        st.stop()
    
    # Verificar API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ No se encontró OPENAI_API_KEY")
        st.markdown("""
        **Para usar el generador, configura tu API Key:**
        
        1. Edita `.streamlit/secrets.toml`:
```toml
        OPENAI_API_KEY = "sk-proj-TU-API-KEY"
```
        
        2. Reinicia la aplicación.
        """)
        st.stop()
    
    # Importar servicio de generación
    try:
        from services.image_generator import (
            generar_imagen,
            mejorar_prompt_educativo,
            obtener_plantilla,
            listar_temas_disponibles,
            listar_conceptos_por_tema,
            sugerir_prompt_educativo,
            PLANTILLAS_PROMPTS
        )
        generator_available = True
    except ImportError as e:
        st.error(f"Error al importar el generador: {e}")
        st.info("Asegúrate de tener el archivo `services/image_generator.py`")
        generator_available = False
        st.stop()
    
    # ========== LAYOUT PRINCIPAL ==========
    col_config, col_preview = st.columns([1.2, 1])
    
    with col_config:
        st.subheader("📝 Configurar Generación")
        
        # Tabs para diferentes modos
        tab_manual, tab_mejorar = st.tabs([
            "✍️ Prompt Manual", 
            "✨ Mejorar con IA"
        ])
        
        # ========== TAB: PROMPT MANUAL ==========
        with tab_manual:
            st.markdown("Escribe tu propio prompt para generar la imagen.")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                tema_manual = st.text_input(
                    "🏷️ Tema",
                    placeholder="Ej: grafos, árboles, pilas...",
                    key="gen_tema_manual"
                )
            with col_t2:
                concepto_manual = st.text_input(
                    "📌 Concepto",
                    placeholder="Ej: BFS, inorden, push...",
                    key="gen_concepto_manual"
                )
            
            prompt_manual = st.text_area(
                "💬 Prompt para GPT-Image 1.5",
                placeholder="Describe la imagen educativa que deseas generar...",
                height=150,
                key="gen_prompt_manual"
            )
            
            st.caption("💡 Tip: Los prompts en inglés suelen dar mejores resultados")
        
                
        # ========== TAB: MEJORAR CON IA ==========
        with tab_mejorar:
            st.markdown("Escribe un prompt básico y GPT-4o lo mejorará para ti.")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                tema_mejorar = st.text_input(
                    "🏷️ Tema",
                    placeholder="Ej: grafos",
                    key="gen_tema_mejorar"
                )
            with col_m2:
                concepto_mejorar = st.text_input(
                    "📌 Concepto",
                    placeholder="Ej: BFS",
                    key="gen_concepto_mejorar"
                )
            
            prompt_basico = st.text_area(
                "💬 Tu prompt básico (en español o inglés)",
                placeholder="Ej: Quiero una imagen que muestre cómo funciona el algoritmo BFS en un grafo, paso a paso",
                height=100,
                key="gen_prompt_basico"
            )
            
            if "prompt_mejorado" not in st.session_state:
                st.session_state.prompt_mejorado = ""
            
            if st.button("✨ Mejorar Prompt con IA", key="btn_mejorar"):
                if prompt_basico.strip():
                    with st.spinner("🤖 GPT-4o está mejorando tu prompt..."):
                        prompt_mejorado = mejorar_prompt_educativo(
                            prompt_basico,
                            tema_mejorar,
                            concepto_mejorar
                        )
                        st.session_state.prompt_mejorado = prompt_mejorado
                else:
                    st.warning("Escribe un prompt básico primero.")
            
            if st.session_state.prompt_mejorado:
                st.markdown("**✅ Prompt mejorado:**")
                prompt_mejorado_edit = st.text_area(
                    "Puedes editarlo antes de generar",
                    value=st.session_state.prompt_mejorado,
                    height=120,
                    key="gen_prompt_mejorado_edit"
                )
        
        # ========== INFORMACIÓN SOBRE GPT-IMAGE 1.5 ==========
        st.markdown("---")
        st.info("📝 **GPT-Image 1.5** genera imágenes de alta calidad optimizadas para contenido educativo")
        
        # ========== DETERMINAR PROMPT FINAL ==========
        prompt_final = ""
        tema_final = ""
        concepto_final = ""

        # Verificar cada fuente de prompt
        if prompt_manual and prompt_manual.strip():
            prompt_final = prompt_manual.strip()
            tema_final = tema_manual
            concepto_final = concepto_manual
        elif 'gen_prompt_mejorado_edit' in st.session_state and st.session_state.get("gen_prompt_mejorado_edit", "").strip():
            prompt_final = st.session_state.get("gen_prompt_mejorado_edit", "").strip()
            tema_final = tema_mejorar
            concepto_final = concepto_mejorar
        elif st.session_state.prompt_mejorado:
            prompt_final = st.session_state.prompt_mejorado
            tema_final = tema_mejorar
            concepto_final = concepto_mejorar

                # ========== BOTÓN DE GENERAR IMAGEN ==========
        st.markdown("---")
        if st.button("🚀 Generar Imagen", type="primary", key="btn_generar"):
            
            if not prompt_final:
                st.error("⚠️ Escribe o selecciona un prompt antes de generar.")
                st.stop()
            
            with st.spinner("🎨 GPT-Image 1.5 está generando tu imagen... (puede tomar 10-30 segundos)"):
                
                resultado = generar_imagen(
                    prompt=prompt_final,
                    size="1024x1024"  # Se ignora pero se pasa por compatibilidad
                )
                
                if resultado["success"]:
                    st.session_state.imagen_generada = {
                        "image_base64": resultado["image_base64"],
                        "revised_prompt": resultado.get("revised_prompt", prompt_final),
                        "prompt_original": prompt_final,
                        "tema": tema_final,
                        "concepto": concepto_final,
                        "modelo": "gpt-image-1.5",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.success("✅ ¡Imagen generada con GPT-Image 1.5!")
                    
                    log_event("image.generate", extra={
                        "tema": tema_final,
                        "concepto": concepto_final,
                        "modelo": "gpt-image-1.5"
                    })
                    rerun()
                else:
                    st.error(f"❌ Error: {resultado.get('error', 'Error desconocido')}")
    
    # ========== COLUMNA DE PREVIEW ==========
    with col_preview:
        st.subheader("👁️ Vista Previa")
        
        if "imagen_generada" in st.session_state and st.session_state.imagen_generada:
            img_data = st.session_state.imagen_generada
            
            # Badge de GPT-Image 1.5
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 12px;
                display: inline-block;
                margin-bottom: 16px;
                font-weight: bold;
            ">
                🟢 GPT-Image 1.5
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar imagen
            import base64
            image_bytes = base64.b64decode(img_data["image_base64"])
            st.image(image_bytes, caption="Imagen generada por GPT-Image 1.5")
            
            with st.expander("📋 Detalles de la generación", expanded=True):
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown(f"**Tema:** {img_data.get('tema', 'N/A')}")
                    st.markdown(f"**Concepto:** {img_data.get('concepto', 'N/A')}")
                    st.markdown(f"**Modelo:** GPT-Image 1.5")
                
                with col_info2:
                    st.markdown(f"**Calidad:** Alta (por defecto)")
                    st.markdown(f"**Generado:** {img_data.get('timestamp', 'N/A')[:19]}")
                
                st.markdown("---")
                st.markdown("**Prompt original:**")
                st.text(img_data.get("prompt_original", "")[:300])
                
                st.markdown("**Prompt usado:**")
                st.text(img_data.get("revised_prompt", "")[:300])
            
            # ========== ACCIONES ==========
            st.markdown("---")
            st.markdown("### 💾 Acciones")
            
            col_act1, col_act2 = st.columns(2)

            from services.supabase_client import supabase

            with col_act1:
                if st.button("📥 Guardar en Catálogo", type="primary", key="btn_guardar_catalogo"):
                    with st.spinner("Guardando y calculando métricas..."):
                        import base64
                        import pandas as pd
                        from uuid import uuid4
                        from datetime import datetime
                        from pathlib import Path
                        from PIL import Image
                        from io import BytesIO

                        # Convertir base64 a bytes
                        image_bytes = base64.b64decode(img_data["image_base64"])

                        # Crear ID único
                        image_id = f"img_{uuid4().hex[:8]}"

                        # Crear carpeta si no existe
                        # Convertir base64 a bytes
                        image_bytes = base64.b64decode(img_data["image_base64"])

                        # Crear ID único
                        image_id = f"img_{uuid4().hex[:8]}"
                        file_name = f"{image_id}.png"

                        # Subir imagen al bucket
                        supabase.storage.from_("imagenes").upload(
                            file_name,
                            image_bytes,
                            {"content-type": "image/png"}
                        )

                        # Obtener URL pública
                        public_url = supabase.storage.from_("imagenes").get_public_url(file_name)

                        # Calcular métricas
                        try:
                            from services.utils import wcag_like_contrast_ratio, edge_density
                            pil_img = Image.open(BytesIO(image_bytes))
                            wcag_val = wcag_like_contrast_ratio(pil_img)
                            edge_val = edge_density(pil_img)
                        except:
                            wcag_val = 0
                            edge_val = 0

                        # Insertar metadata en PostgreSQL
                        supabase.table("imagenes").insert({
                            "image_id": image_id,
                            "tema": img_data.get("tema", ""),
                            "concepto": img_data.get("concepto", ""),
                            "prompt": img_data.get("prompt_original", ""),
                            "image_url": public_url,
                            "herramienta": "gpt-image-1.5",
                            "wcag_ratio": round(wcag_val, 2),
                            "edge_density": round(edge_val, 3),
                            "estado": "pendiente",
                            "version": "v1",
                            "autor": user_name,
                            "alt_text": img_data.get("revised_prompt", "")[:200],
                            "fecha": datetime.now().isoformat()
                        }).execute()

                        st.success(f"✅ Imagen guardada en Supabase | WCAG: {wcag_val:.2f}")

                        
                        
                        # Limpiar sesión
                        st.session_state.imagen_generada = None
                        rerun()

            with col_act2:
                if st.button("🗑️ Descartar", key="btn_descartar"):
                    st.session_state.imagen_generada = None
                    rerun()
            
            # ========== REGENERAR ==========
            st.markdown("---")
            if st.button("🔄 Regenerar", key="btn_regenerar"):
                with st.spinner("🎨 Regenerando con GPT-Image 1.5..."):
                    resultado = generar_imagen(
                        prompt=img_data["prompt_original"],
                        size="1024x1024"
                    )
                    
                    if resultado["success"]:
                        st.session_state.imagen_generada = {
                            **img_data,
                            "image_base64": resultado["image_base64"],
                            "timestamp": datetime.now().isoformat()
                        }
                        st.success("✅ ¡Nueva imagen generada!")
                        rerun()
                    else:
                        st.error(f"❌ Error: {resultado.get('error', 'Error desconocido')}")
        
        # ========== SIN IMAGEN GENERADA ==========
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 16px;
                padding: 60px 30px;
                text-align: center;
                color: #94a3b8;
                border: 2px dashed #475569;
            ">
                <div style="font-size: 4rem; margin-bottom: 20px;">🖼️</div>
                <h3 style="color: #e2e8f0; margin: 0;">Sin imagen generada</h3>
                <p style="margin-top: 10px;">
                    Configura un prompt y haz clic en<br>
                    <strong>"🚀 Generar Imagen"</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Tips para GPT-Image 1.5")
            st.markdown("""
            - **Sé específico:** Describe exactamente qué quieres
            - **Usa inglés:** Mejores resultados técnicos
            - **Menciona estilo:** "educational diagram", "professional"
            - **Incluye colores:** Ayuda a la claridad visual
            """)
                        

# -------------------------------------------------------
# EVALUACIÓN IA CON CHAT INTERACTIVO (ACTUALIZADO)
# -------------------------------------------------------

elif seccion == "Evaluación IA (imagen + prompt)":
    st.header("🤖 Evaluación IA con Chat Interactivo")
    
    # Verificar configuración de OpenAI
    try:
        llm_ready = is_llm_ready() if callable(is_llm_ready) else False
    except:
        llm_ready = False
    
    # Mostrar estado de configuración
    if not llm_ready:
        st.warning("⚠️ OpenAI no está configurado correctamente.")
        st.markdown("""
        **Para activar la evaluación con IA:**
        
        1. Edita `.streamlit/secrets.toml`:
        ```toml
        LLM_ENABLE = "1"
        LLM_PROVIDER = "openai"
        OPENAI_API_KEY = "sk-proj-TU-API-KEY"
        LLM_MODEL = "gpt-4o"
        ```
        
        2. Reinicia la aplicación.
        """)
        st.stop()
    
    st.success("✅ OpenAI conectado correctamente")
    
    # ✅ CÓDIGO CORREGIDO
    try:
        response = supabase.table("imagenes").select("*").execute()
        imagenes_data = response.data
        if not imagenes_data:
            st.warning("No hay imágenes en Supabase. Genera algunas imágenes primero.")
            st.stop()
        catalogo = pd.DataFrame(imagenes_data)  # ← Dentro del try, donde debe estar
    except Exception as e:
        st.error(f"❌ Error al obtener datos de Supabase: {str(e)}")
        st.stop()

        # Convertir a DataFrame
        catalogo = pd.DataFrame(response.data)
    
    if len(catalogo) == 0:
        st.warning("No hay imágenes en el catálogo. Genera algunas imágenes primero.")
        st.stop()
    
    # ========== COLUMNA IZQUIERDA: Selector de imagen ==========
    col_img, col_chat = st.columns([1, 1.5])

    with col_img:
        st.subheader("📷 Seleccionar Imagen")
        
        # ===== BUSCADOR POR TEMAS =====
        temas_disponibles = ["(Todos)"] + sorted(catalogo["tema"].dropna().unique().tolist())
        
        col_filtro1, col_filtro2 = st.columns(2)
        with col_filtro1:
            tema_filtro = st.selectbox(
                "🏷️ Filtrar por tema",
                temas_disponibles,
                key="ia_filtro_tema"
            )
        
        with col_filtro2:
            buscar_texto = st.text_input(
                "🔍 Buscar",
                placeholder="concepto, ID...",
                key="ia_buscar_texto"
            )
        
        # Aplicar filtros
        catalogo_filtrado = catalogo.copy()
        
        if tema_filtro != "(Todos)":
            catalogo_filtrado = catalogo_filtrado[catalogo_filtrado["tema"] == tema_filtro]
        
        if buscar_texto.strip():
            mask = (
                catalogo_filtrado["image_id"].str.contains(buscar_texto, case=False, na=False) |
                catalogo_filtrado["concepto"].str.contains(buscar_texto, case=False, na=False) |
                catalogo_filtrado["prompt"].fillna("").str.contains(buscar_texto, case=False, na=False)
            )
            catalogo_filtrado = catalogo_filtrado[mask]
        
        # Mostrar contador de resultados
        st.caption(f"📊 {len(catalogo_filtrado)} imágenes encontradas")
        
        if catalogo_filtrado.empty:
            st.warning("No hay imágenes que coincidan con los filtros.")
            st.stop()
        
        # Selector de imagen
        image_ids = catalogo_filtrado["image_id"].tolist()
        
        # Detectar cambio de imagen para resetear chat
        if "ia_current_image" not in st.session_state:
            st.session_state.ia_current_image = image_ids[0] if image_ids else None
        
        # Si la imagen actual ya no está en el filtro, resetear
        if st.session_state.ia_current_image not in image_ids:
            st.session_state.ia_current_image = image_ids[0] if image_ids else None
            st.session_state.ia_chat_history = []
            st.session_state.ia_analysis_done = False
        
        # Crear opciones con más contexto
        opciones_imagen = {
            img_id: f"{img_id} | {catalogo_filtrado[catalogo_filtrado['image_id']==img_id]['concepto'].values[0]}"
            for img_id in image_ids
        }
        
        selected_id = st.selectbox(
            "Imagen del catálogo",
            image_ids,
            index=image_ids.index(st.session_state.ia_current_image) if st.session_state.ia_current_image in image_ids else 0,
            format_func=lambda x: opciones_imagen.get(x, x),
            key="ia_image_selector"
        )
        
        # Si cambió la imagen, resetear el chat
        if selected_id != st.session_state.ia_current_image:
            st.session_state.ia_current_image = selected_id
            st.session_state.ia_chat_history = []
            st.session_state.ia_analysis_done = False
            rerun()
        
        # ✅ Leer directo del catalogo de Supabase
        row = catalogo[catalogo["image_id"] == selected_id].iloc[0]
        
        # ✅ CORREGIDO - usa image_url de Supabase
        image_url = row.get("image_url", "")
        if image_url:
            st.image(image_url, caption=f"{row['tema']} / {row['concepto']}")
        else:
            st.error("❌ Imagen no encontrada en el bucket")
            st.stop()
        
        # Información de la imagen
        st.markdown("---")
        st.markdown("**📋 Información:**")
        st.write(f"**Tema:** {row['tema']}")
        st.write(f"**Concepto:** {row['concepto']}")
        
        # 🆕 MOSTRAR HERRAMIENTA CON BADGE PARA GPT-IMAGE 1.5
        herramienta = row.get('herramienta', 'N/A')
        if 'gpt-image' in str(herramienta).lower():
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 4px 12px;
                border-radius: 8px;
                display: inline-block;
                font-size: 0.85rem;
                font-weight: bold;
                margin-top: 8px;
            ">
                🟢 Generado con GPT-Image 1.5
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write(f"**Herramienta:** {herramienta}")
        
        # Mostrar prompt en un expander
        with st.expander("📝 Ver Prompt usado", expanded=True):
            prompt_actual = row.get('prompt', 'Sin prompt')
            if pd.isna(prompt_actual) or str(prompt_actual).strip() == '':
                prompt_actual = 'Sin prompt disponible'
            st.text_area(
                "Prompt",
                value=str(prompt_actual),
                height=120,
                disabled=True,
                key=f"ia_prompt_display_{selected_id}",
                label_visibility="collapsed"
            )
        
        # Métricas técnicas
        with st.expander("📊 Métricas técnicas"):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Contraste", f"{row.get('wcag_ratio', 'N/A')}")
            with col_m2:
                st.metric("Densidad bordes", f"{row.get('edge_density', 'N/A')}")
    
    # ========== COLUMNA DERECHA: Chat interactivo ==========
    with col_chat:
        st.subheader("💬 Chat con IA")
        
        # Inicializar historial de chat si no existe
        if "ia_chat_history" not in st.session_state:
            st.session_state.ia_chat_history = []
        
        if "ia_analysis_done" not in st.session_state:
            st.session_state.ia_analysis_done = False
        
        # Botón para iniciar análisis
        if not st.session_state.ia_analysis_done:
            st.info("👆 Haz clic en el botón para que la IA analice esta imagen")
            
            if st.button("🔍 Iniciar Análisis con IA", type="primary"):
                with st.spinner("🤖 Analizando imagen y prompt..."):
                    try:
                        from services.openai_eval import generar_analisis_inicial
                        
                        # Generar análisis inicial
                        analisis = generar_analisis_inicial(
                            tema=row['tema'],
                            concepto=row['concepto'],
                            prompt_imagen=row.get('prompt', ''),
                            imagen_path=image_url
                        )
                        
                        # Guardar en historial
                        st.session_state.ia_chat_history.append({
                            "role": "assistant",
                            "content": analisis
                        })
                        st.session_state.ia_analysis_done = True
                        
                        log_event("ia_chat.start", ref_id=selected_id, extra={
                            "tema": row['tema'],
                            "concepto": row['concepto']
                        })
                        
                        rerun()
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        # Mostrar solo mensaje amigable
                        if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
                            st.error("⚠️ Se acabaron los créditos de OpenAI")
                            st.info("💡 Recarga tu cuenta en: https://platform.openai.com/billing")
                        elif "rate_limit" in error_str or "429" in error_str:
                            st.warning("⏳ Demasiadas solicitudes. Espera 1-2 minutos e intenta de nuevo.")
                        elif "invalid_api_key" in error_str or "401" in error_str:
                            st.error("🔑 API Key inválida. Verifica tu configuración.")
                        else:
                            st.error(f"❌ Error: {error_str[:150]}")
        
        else:
            # Mostrar historial de chat
            chat_container = st.container()
            
            with chat_container:
                for i, msg in enumerate(st.session_state.ia_chat_history):
                    if msg["role"] == "assistant":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(msg["content"])
            
            st.markdown("---")
            
            # Input para nueva pregunta
            pregunta = st.chat_input(
                "Escribe tu pregunta sobre la imagen...",
                key="ia_chat_input"
            )
            
            if pregunta:
                st.session_state.ia_chat_history.append({
                    "role": "user",
                    "content": pregunta
                })
                
                with st.spinner("🤖 Pensando..."):
                    try:
                        from services.openai_eval import chat_evaluacion_ia
                        
                        respuesta = chat_evaluacion_ia(
                            mensaje_usuario=pregunta,
                            historial=st.session_state.ia_chat_history[:-1],
                            tema=row['tema'],
                            concepto=row['concepto'],
                            prompt_imagen=row.get('prompt', ''),
                            imagen_path=image_url
                        )
                        
                        st.session_state.ia_chat_history.append({
                            "role": "assistant",
                            "content": respuesta
                        })
                        
                        log_event("ia_chat.message", ref_id=selected_id, extra={
                            "pregunta": pregunta[:100]
                        })
                        
                        rerun()
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
                            st.error("⚠️ Se acabaron los créditos de OpenAI")
                            st.info("💡 Recarga en: https://platform.openai.com/billing")
                        elif "rate_limit" in error_str or "429" in error_str:
                            st.warning("⏳ Espera 1-2 minutos e intenta de nuevo.")
                        else:
                            st.error(f"❌ Error: {error_str[:150]}")
            
            # Sugerencias de preguntas
            st.markdown("---")
            st.markdown("**💡 Sugerencias de preguntas:**")
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                if st.button("¿Hay errores conceptuales?", key="sug1"):
                    st.session_state.ia_pregunta_sugerida = "¿Detectas algún error conceptual en la imagen que pueda confundir a los estudiantes?"
                    rerun()
                
                if st.button("¿Cómo mejorar el prompt?", key="sug2"):
                    st.session_state.ia_pregunta_sugerida = "¿Cómo mejorarías el prompt para generar una imagen más clara y didáctica?"
                    rerun()
            
            with col_s2:
                if st.button("¿Es útil para enseñar?", key="sug3"):
                    st.session_state.ia_pregunta_sugerida = "¿Esta imagen es útil para enseñar el concepto a estudiantes universitarios? ¿Por qué?"
                    rerun()
                
                if st.button("Dame una puntuación", key="sug4"):
                    st.session_state.ia_pregunta_sugerida = "Dame una puntuación del 1-5 para coherencia, fidelidad y claridad, explicando cada una."
                    rerun()
            
            # Procesar pregunta sugerida
            if "ia_pregunta_sugerida" in st.session_state and st.session_state.ia_pregunta_sugerida:
                pregunta_sug = st.session_state.ia_pregunta_sugerida
                st.session_state.ia_pregunta_sugerida = ""
                
                st.session_state.ia_chat_history.append({
                    "role": "user",
                    "content": pregunta_sug
                })
                
                with st.spinner("🤖 Pensando..."):
                    try:
                        from services.openai_eval import chat_evaluacion_ia
                        
                        respuesta = chat_evaluacion_ia(
                            mensaje_usuario=pregunta_sug,
                            historial=st.session_state.ia_chat_history[:-1],
                            tema=row['tema'],
                            concepto=row['concepto'],
                            prompt_imagen=row.get('prompt', ''),
                            imagen_path=image_url
                        )
                        
                        st.session_state.ia_chat_history.append({
                            "role": "assistant",
                            "content": respuesta
                        })
                        
                        rerun()
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
                            st.error("⚠️ Se acabaron los créditos de OpenAI")
                            st.info("💡 Recarga en: https://platform.openai.com/billing")
                        elif "rate_limit" in error_str or "429" in error_str:
                            st.warning("⏳ Espera 1-2 minutos e intenta de nuevo.")
                        else:
                            st.error(f"❌ Error: {error_str[:150]}")
            
            # Botones de acción
            st.markdown("---")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("🗑️ Limpiar Chat"):
                    st.session_state.ia_chat_history = []
                    st.session_state.ia_analysis_done = False
                    rerun()
            
            with col_btn2:
                if st.button("💾 Guardar Evaluación"):
                    try:
                        # Crear conversación completa
                        conversacion_completa = ""
                        for msg in st.session_state.ia_chat_history:
                            if msg["role"] == "assistant":
                                conversacion_completa += f"🤖 IA: {msg['content']}\n\n---\n\n"
                            else:
                                conversacion_completa += f"👤 Usuario: {msg['content']}\n\n---\n\n"
                        
                        # Extraer puntuaciones
                        coherencia = 0
                        fidelidad = 0
                        claridad = 0
                        
                        import re
                        for msg in st.session_state.ia_chat_history:
                            contenido = msg['content'].lower()
                            
                            match = re.search(r'coherencia[*\s:]*(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?', contenido)
                            if match:
                                coherencia = float(match.group(1))
                            
                            match = re.search(r'fidelidad[*\s:]*(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?', contenido)
                            if match:
                                fidelidad = float(match.group(1))
                            
                            match = re.search(r'claridad[*\s:]*(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?', contenido)
                            if match:
                                claridad = float(match.group(1))
                        
                        # Guardar en CSV
                        ia_eval_dir = Path("data/ia_evaluations")
                        ia_eval_dir.mkdir(parents=True, exist_ok=True)
                        ia_eval_path = ia_eval_dir / "ia_eval.csv"
                        
                        nueva_eval = {
                            "image_id": selected_id,
                            "tema": row['tema'],
                            "concepto": row['concepto'],
                            "prompt": row.get('prompt', ''),
                            "herramienta": row.get('herramienta', 'N/A'),  # 🆕 Incluir herramienta
                            "coherencia": coherencia,
                            "fidelidad": fidelidad,
                            "claridad": claridad,
                            "errores": "Ver conversación completa",
                            "recomendaciones": conversacion_completa,
                            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        
                        if ia_eval_path.exists():
                            df_eval = pd.read_csv(ia_eval_path, encoding="utf-8")
                            df_eval = df_eval[df_eval["image_id"] != selected_id]
                            df_eval = pd.concat([df_eval, pd.DataFrame([nueva_eval])], ignore_index=True)
                        else:
                            df_eval = pd.DataFrame([nueva_eval])
                        
                        df_eval.to_csv(ia_eval_path, index=False, encoding="utf-8")

                        # ----- Guardar en Supabase -----
                        try:
                            supabase.table("evaluaciones_ia").insert({
                                "image_id": nueva_eval["image_id"],
                                "tema": nueva_eval["tema"],
                                "concepto": nueva_eval["concepto"],
                                "prompt": nueva_eval["prompt"],
                                "herramienta": nueva_eval["herramienta"],
                                "coherencia": nueva_eval["coherencia"],
                                "fidelidad": nueva_eval["fidelidad"],
                                "claridad": nueva_eval["claridad"],
                                "errores": nueva_eval["errores"],
                                "recomendaciones": nueva_eval["recomendaciones"],
                                "fecha": nueva_eval["fecha"]  # usa el mismo timestamp que CSV
                            }).execute()
                            st.success("✅ Evaluación también guardada en Supabase")
                        except Exception as e:
                            st.error(f"⚠️ No se pudo guardar en Supabase: {e}")
                        
                        log_event("ia_chat.save", ref_id=selected_id, extra={
                            "mensajes": len(st.session_state.ia_chat_history),
                            "coherencia": coherencia,
                            "fidelidad": fidelidad,
                            "claridad": claridad
                        })
                                                
                    except Exception as e:
                        st.error(f"❌ Error al guardar: {str(e)}")
            
            with col_btn3:
                if st.button("📥 Exportar Chat"):
                    try:
                        export_text = f"""# Evaluación IA - {selected_id}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Tema: {row['tema']}
Concepto: {row['concepto']}
Herramienta: {row.get('herramienta', 'N/A')}
Prompt: {row.get('prompt', 'N/A')}

## Conversación:
"""
                        for msg in st.session_state.ia_chat_history:
                            role = "🤖 IA" if msg["role"] == "assistant" else "👤 Usuario"
                            export_text += f"\n### {role}:\n{msg['content']}\n"
                        
                        export_path = DATA_DIR / f"ia_chat_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        with open(export_path, "w", encoding="utf-8") as f:
                            f.write(export_text)
                        
                        st.success(f"✅ Chat exportado a: {export_path.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error al exportar: {str(e)}")

# -------------------------------------------------------
# Galería MEJORADA con vista detallada modal
# Reemplaza la sección "elif seccion == 'Galería':" en tu app.py
# -------------------------------------------------------

elif seccion == "Galería":

    try:
        response = supabase.table("imagenes").select("*").execute()
        df_meta = pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error al cargar imágenes: {e}")
        df_meta = pd.DataFrame()
    # ========== CSS PARA GALERÍA MEJORADA ==========
    st.markdown("""
    <style>
    /* ===== GALERÍA GRID ===== */
    .gallery-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 20px;
        padding: 10px 0;
    }
    
    /* ===== TARJETA DE IMAGEN ===== */
    .image-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .image-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.5);
    }
                
    /* 🆕 AÑADIR ESTILOS PARA BADGE GPT-IMAGE */
    .badge-gpt-image {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .herramienta-highlight {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .image-card img {
        width: 100%;
        height: auto;
        min-height: 180px;
        max-height: 220px;
        object-fit: contain;
        background: #000;
        transition: transform 0.3s ease;
        display: block;
    }
    
    .image-card:hover img {
        transform: scale(1.05);
    }
    
    .card-content {
        padding: 16px;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 12px;
    }
    
    .meta-tag {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .meta-tag.herramienta {
        background: rgba(168, 85, 247, 0.2);
        color: #c4b5fd;
    }
    
    .meta-tag.version {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
    }
    
    /* Estados de la imagen */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-pendiente {
        background: rgba(251, 191, 36, 0.2);
        color: #fcd34d;
    }
    
    .status-aceptar {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
    }
    
    .status-ajustar {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
    }
    
    .status-descartar {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }
    
    /* ===== VISTA DETALLE (MODAL) ===== */
    .detail-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 25px 80px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .detail-image-container {
        position: relative;
        background: #000;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 400px;
        max-height: 500px;
        overflow: hidden;
    }
    
    .detail-image-container img {
        max-width: 100%;
        max-height: 500px;
        object-fit: contain;
    }
    
    .detail-content {
        padding: 32px;
    }
    
    .detail-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .detail-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    
    .detail-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    .detail-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #64748b;
        background: rgba(100,116,139,0.2);
        padding: 6px 12px;
        border-radius: 8px;
    }
    
    /* Info Grid */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .info-item {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .info-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .info-value {
        font-size: 1.1rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .info-value.highlight {
        color: #60a5fa;
    }
    
    /* Prompt Section */
    .prompt-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .prompt-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }
    
    .prompt-icon {
        font-size: 1.5rem;
    }
    
    .prompt-title {
        font-size: 1rem;
        font-weight: 600;
        color: #93c5fd;
    }
    
    .prompt-text {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.6;
        background: rgba(0,0,0,0.3);
        padding: 16px;
        border-radius: 10px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* Metrics */
    .metrics-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fbbf24;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Alt Text */
    .alt-text-box {
        background: rgba(34, 197, 94, 0.1);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #22c55e;
    }
    
    .alt-text-label {
        font-size: 0.8rem;
        color: #86efac;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .alt-text-content {
        color: #d1fae5;
        font-style: italic;
        line-height: 1.5;
    }
    
    /* Back button */
    .back-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.1);
        color: #f1f5f9;
        padding: 10px 20px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid rgba(255,255,255,0.1);
        cursor: pointer;
    }
    
    .back-btn:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(-4px);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    /* Paginación mejorada */
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin: 20px 0;
        padding: 16px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
    }
    
    .page-info {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    .page-info strong {
        color: #f1f5f9;
    }
    
    /* Filtros */
    .filters-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .filter-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Empty state */
    .empty-gallery {
        text-align: center;
        padding: 80px 20px;
        color: #64748b;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    
    .empty-text {
        font-size: 1.1rem;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🖼️ Galería de Imágenes")
    
    if df_meta.empty:
        st.markdown("""
        <div class="empty-gallery">
            <div class="empty-icon">🖼️</div>
            <div class="empty-text">No hay imágenes en el catálogo todavía.<br>
            ¡Genera algunas imágenes para comenzar!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ========== ESTADO PARA VISTA DETALLE ==========
        if "gallery_view_image" not in st.session_state:
            st.session_state.gallery_view_image = None
        
        # ========== VISTA DETALLE DE IMAGEN ==========
        if st.session_state.gallery_view_image is not None:
            img_id = st.session_state.gallery_view_image
            img_data = df_meta[df_meta["image_id"] == img_id]
            
            if img_data.empty:
                st.error("Imagen no encontrada")
                st.session_state.gallery_view_image = None
                rerun()
            else:
                row = img_data.iloc[0]
                
                # Botón volver
                if st.button("← Volver a la Galería", key="btn_back_gallery"):
                    st.session_state.gallery_view_image = None
                    rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Layout principal: Imagen + Info
                col_img, col_info = st.columns([1.2, 1])
                
                with col_img:
                    # Contenedor de imagen
                    image_url = row.get("image_url", "")
                    if image_url:
                        st.image(image_url)
                    else:
                        st.error("❌ Imagen no encontrada en el bucket")
                    
                    # 🆕 BADGE GPT-IMAGE debajo de la imagen
                    herramienta = row.get('herramienta', '')
                    if 'gpt-image' in str(herramienta).lower():
                        st.markdown("""
                        <div style="text-align: center; margin-top: 12px;">
                            <span class="badge-gpt-image">
                                🟢 Generado con GPT-Image 1.5
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Métricas debajo de la imagen
                    st.markdown("### 📊 Métricas Técnicas")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        wcag = row.get('wcag_ratio', None)
                        # 🔒 Protección robusta contra NaN
                        if wcag is not None and not pd.isna(wcag):
                            try:
                                wcag_val = float(wcag)
                                color = "🟢" if wcag_val >= 4.5 else ("🟡" if wcag_val >= 3 else "🔴")
                                st.metric(f"{color} Contraste WCAG", f"{wcag_val:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Contraste WCAG", "Sin calcular")
                        else:
                            st.metric("Contraste WCAG", "Sin calcular")
                    
                    with metric_cols[1]:
                        edge = row.get('edge_density', None)
                        # 🔒 Protección robusta contra NaN
                        if edge is not None and not pd.isna(edge):
                            try:
                                edge_val = float(edge)
                                st.metric("📐 Densidad Bordes", f"{edge_val:.3f}")
                            except (ValueError, TypeError):
                                st.metric("Densidad Bordes", "Sin calcular")
                        else:
                            st.metric("Densidad Bordes", "Sin calcular")
                    
                    with metric_cols[2]:
                        estado = row.get('estado', 'pendiente')
                        if pd.isna(estado) or not isinstance(estado, str):
                            estado = "pendiente"
                        estado = estado.strip().lower()
                        
                        emoji_estado = {
                            'pendiente': '⏳',
                            'aceptar': '✅',
                            'ajustar': '🔧',
                            'descartar': '❌'
                        }.get(estado, '❓')
                        
                        st.metric(f"{emoji_estado} Estado", estado.capitalize())
                
                with col_info:
                    # Header con ID y estado
                    estado = row.get('estado', 'pendiente')
                    estado_colors = {
                        'pendiente': '#fbbf24',
                        'aceptar': '#22c55e',
                        'ajustar': '#3b82f6',
                        'descartar': '#ef4444'
                    }
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                                border-radius: 16px; padding: 24px; margin-bottom: 20px;
                                border: 1px solid rgba(255,255,255,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <span style="font-family: monospace; font-size: 0.85rem; color: #64748b; 
                                        background: rgba(100,116,139,0.2); padding: 6px 12px; border-radius: 8px;">
                                {row.get('image_id', 'N/A')}
                            </span>
                            <span style="background: {estado_colors.get(estado, '#64748b')}22; 
                                        color: {estado_colors.get(estado, '#64748b')}; 
                                        padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
                                        text-transform: uppercase;">
                                {estado}
                            </span>
                        </div>
                        <h2 style="color: #f8fafc; margin: 0 0 8px 0; font-size: 1.6rem;">
                            {row.get('tema', 'Sin tema').capitalize()}
                        </h2>
                        <p style="color: #94a3b8; margin: 0; font-size: 1.1rem;">
                            📌 {row.get('concepto', 'Sin concepto')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 🆕 Información en grid CON HIGHLIGHT PARA GPT-IMAGE
                    st.markdown("### 📋 Información")
                    
                    info_col1, info_col2 = st.columns(2)
                    
                    herramienta_val = row.get('herramienta', 'N/A')
                    is_gpt_image = 'gpt-image' in str(herramienta_val).lower()
                    herramienta_class = "herramienta-highlight" if is_gpt_image else ""
                    
                    with info_col1:
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 14px; margin-bottom: 12px;"
                             class="{herramienta_class}">
                            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">🛠️ Herramienta</div>
                            <div style="font-size: 1rem; color: {'#10b981' if is_gpt_image else '#e2e8f0'}; font-weight: 500; margin-top: 4px;">
                                {herramienta_val}
                                {'<span style="margin-left: 6px;">🟢</span>' if is_gpt_image else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">📅 Fecha</div>
                            <div style="font-size: 1rem; color: #e2e8f0; font-weight: 500; margin-top: 4px;">{row.get('fecha', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with info_col2:
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">🏷️ Versión</div>
                            <div style="font-size: 1rem; color: #e2e8f0; font-weight: 500; margin-top: 4px;">{row.get('version', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">👤 Autor</div>
                            <div style="font-size: 1rem; color: #e2e8f0; font-weight: 500; margin-top: 4px;">{row.get('autor', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Prompt
                    prompt = row.get('prompt', '')
                    if prompt and str(prompt).strip() and str(prompt).lower() != 'nan':
                        st.markdown("### 💬 Prompt Usado")
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                                    border-radius: 12px; padding: 16px; border: 1px solid rgba(59, 130, 246, 0.2);">
                            <div style="font-family: 'Fira Code', monospace; font-size: 0.85rem; color: #cbd5e1; 
                                        line-height: 1.6; white-space: pre-wrap; word-break: break-word;">
{prompt}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Texto alternativo
                    alt_text = row.get('alt_text', '')
                    if alt_text and str(alt_text).strip() and str(alt_text).lower() != 'nan':
                        st.markdown("### ♿ Texto Alternativo")
                        st.markdown(f"""
                        <div style="background: rgba(34, 197, 94, 0.1); border-radius: 12px; padding: 16px;
                                    border-left: 4px solid #22c55e;">
                            <div style="color: #d1fae5; font-style: italic; line-height: 1.5;">
                                "{alt_text}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ========== EVALUACIÓN IA (tu código actual) ==========
                st.markdown("---")
                st.markdown("### 🤖 Evaluación IA")
                
                try:
                        eval_response = supabase.table("evaluaciones_ia").select("*").eq("image_id", img_id).execute()
                        eval_imagen = pd.DataFrame(eval_response.data) if eval_response.data else pd.DataFrame()
                        
                        if not eval_imagen.empty:
                            eval_row = eval_imagen.iloc[-1]
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
                                        border-radius: 16px; padding: 20px; margin-bottom: 16px;
                                        border: 1px solid rgba(139, 92, 246, 0.2);">
                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                    <span style="font-size: 1.5rem;">✅</span>
                                    <span style="color: #a78bfa; font-weight: 600; font-size: 1.1rem;">Imagen Evaluada por IA</span>
                                </div>
                                <div style="color: #64748b; font-size: 0.85rem;">
                                    📅 Evaluada el: {eval_row.get('fecha', 'N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Puntuaciones IA
                            # --- PUNTUACIONES IA ---
                            coherencia = eval_row.get('coherencia', 0)
                            fidelidad = eval_row.get('fidelidad', 0)
                            claridad = eval_row.get('claridad', 0)

                            # Mostrar métricas en HTML (sin columnas anidadas)
                            st.markdown(f"""
                            <div style="display: flex; gap: 20px; margin: 16px 0;">
                                <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                    <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Coherencia</div>
                                    <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{coherencia}/5</div>
                                </div>
                                <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                    <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Fidelidad</div>
                                    <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{fidelidad}/5</div>
                                </div>
                                <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                    <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Claridad</div>
                                    <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{claridad}/5</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Errores detectados
                            errores = eval_row.get('errores', '')
                            if errores and str(errores).strip() and str(errores).lower() not in ['nan', 'ver conversación completa']:
                                with st.expander("⚠️ Errores Detectados", expanded=False):
                                    st.markdown(f"""
                                    <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 14px;
                                                border-left: 3px solid #ef4444;">
                                        <div style="color: #fca5a5; line-height: 1.6;">{errores}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Recomendaciones
                            recomendaciones = eval_row.get('recomendaciones', '')
                            if recomendaciones and str(recomendaciones).strip() and str(recomendaciones).lower() != 'nan':
                                with st.expander("💡 Ver conversación completa con IA", expanded=False):
                                    st.markdown(f"""
                                    <div style="background: rgba(34, 197, 94, 0.1); border-radius: 10px; padding: 14px;
                                                border-left: 3px solid #22c55e; max-height: 300px; overflow-y: auto;">
                                        <div style="color: #86efac; line-height: 1.6; white-space: pre-wrap; font-size: 0.9rem;">{recomendaciones[:2000]}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown("""
                            <div style="background: rgba(251, 191, 36, 0.1); border-radius: 12px; padding: 20px;
                                        border: 1px dashed rgba(251, 191, 36, 0.3); text-align: center;">
                                <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
                                <div style="color: #fcd34d; font-weight: 500;">Sin evaluación IA</div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">
                                    Ve a "Evaluación IA" para analizar esta imagen
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error al cargar evaluaciones: {e}")
                                           
                # Acciones (solo docente) - TU CÓDIGO ORIGINAL
                if is_docente:
                    st.markdown("---")
                    st.markdown("### ⚙️ Acciones")
                    
                    action_cols = st.columns([1, 1, 1, 2])
                    
                    with action_cols[0]:
                        if st.button("✅ Aceptar", key="detail_aceptar"):
                            supabase.table("imagenes").update({"estado": "aceptar"}).eq("image_id", img_id).execute()
                            st.success("Estado actualizado a 'aceptar'")
                            log_event("catalog.update_state", ref_id=img_id, extra={"estado": "aceptar"})
                            rerun()
                    
                    with action_cols[1]:
                        if st.button("🔧 Ajustar", key="detail_ajustar"):
                            supabase.table("imagenes").update({"estado": "ajustar"}).eq("image_id", img_id).execute()
                            st.success("Estado actualizado a 'ajustar'")
                            log_event("catalog.update_state", ref_id=img_id, extra={"estado": "ajustar"})
                            rerun()
                    
                    with action_cols[2]:
                        if st.button("❌ Descartar", key="detail_descartar"):
                            supabase.table("imagenes").update({"estado": "descartar"}).eq("image_id", img_id).execute()
                            st.warning("Estado actualizado a 'descartar'")
                            log_event("catalog.update_state", ref_id=img_id, extra={"estado": "descartar"})
                            rerun()
                    
                    with action_cols[3]:
                        if st.button("🗑️ Eliminar permanentemente", key="detail_eliminar", type="secondary"):
                            st.session_state.confirm_delete_image = img_id
                    
                    # Confirmación de eliminación
                    if st.session_state.get("confirm_delete_image") == img_id:
                        st.warning("⚠️ ¿Estás seguro de eliminar esta imagen permanentemente?")
                        del_cols = st.columns([1, 1, 3])
                        with del_cols[0]:
                            if st.button("Sí, eliminar", key="confirm_del_yes", type="primary"):
                                try:
                                    # Eliminar del bucket
                                    file_name = f"{img_id}.png"
                                    supabase.storage.from_("imagenes").remove([file_name])
                                    # Eliminar de la tabla
                                    supabase.table("imagenes").delete().eq("image_id", img_id).execute()
                                    log_event("catalog.delete", ref_id=img_id)
                                    st.success("Imagen eliminada")
                                    st.session_state.gallery_view_image = None
                                    st.session_state.confirm_delete_image = None
                                    rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        with del_cols[1]:
                            if st.button("Cancelar", key="confirm_del_no"):
                                st.session_state.confirm_delete_image = None
                                rerun()
        
        # ========== VISTA DE GALERÍA (GRID) - TU CÓDIGO ORIGINAL ==========
        else:
            # Filtros
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
                        border-radius: 16px; padding: 20px; margin-bottom: 24px;
                        border: 1px solid rgba(255,255,255,0.08);">
                <h4 style="color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
                    🔍 Filtros
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            filter_cols = st.columns([1, 1, 1, 2])
            
            with filter_cols[0]:
                f_tema = st.selectbox(
                    "🏷️ Tema",
                    ["(todos)"] + sorted(df_meta["tema"].dropna().astype(str).unique().tolist()),
                    key="gal_f_tema"
                )
            
            with filter_cols[1]:
                f_herr = st.selectbox(
                    "🛠️ Herramienta",
                    ["(todas)"] + sorted(df_meta["herramienta"].dropna().astype(str).unique().tolist()),
                    key="gal_f_herr"
                )
            
            with filter_cols[2]:
                f_estado = st.selectbox(
                    "📊 Estado",
                    ["(todos)", "pendiente", "aceptar", "ajustar", "descartar"],
                    key="gal_f_estado"
                )
            
            with filter_cols[3]:
                search_text = st.text_input(
                    "🔎 Buscar por tema o concepto",
                    placeholder="Escribe para buscar...",
                    key="gal_search"
                )
            
            # Aplicar filtros
            view = df_meta.copy()
            
            if f_tema != "(todos)":
                view = view[view["tema"] == f_tema]
            if f_herr != "(todas)":
                view = view[view["herramienta"] == f_herr]
            if f_estado != "(todos)":
                view = view[view["estado"] == f_estado]
            if search_text.strip():
                mask = (
                    view["tema"].fillna("").astype(str).str.contains(search_text, case=False, na=False) |
                    view["concepto"].fillna("").astype(str).str.contains(search_text, case=False, na=False)
                )
                view = view[mask]
            
            if view.empty:
                st.markdown("""
                <div class="empty-gallery">
                    <div class="empty-icon">🔍</div>
                    <div class="empty-text">No hay imágenes que coincidan con los filtros.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Ordenar y resetear índice
                view = view.sort_values(["fecha", "tema", "concepto"], ascending=[False, True, True])
                view = view.reset_index(drop=True)
                
                # Paginación
                PAGE_SIZE = 9
                total_items = len(view)
                total_pages = max(1, math.ceil(total_items / PAGE_SIZE))
                
                if "gal_page" not in st.session_state:
                    st.session_state.gal_page = 1
                
                page = max(1, min(st.session_state.gal_page, total_pages))
                st.session_state.gal_page = page
                
                start = (page - 1) * PAGE_SIZE
                end = start + PAGE_SIZE
                page_df = view.iloc[start:end]
                
                # Info de resultados
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <span style="color: #64748b; font-size: 0.9rem;">
                        Mostrando <strong style="color: #f1f5f9;">{start+1}-{min(end, total_items)}</strong> de 
                        <strong style="color: #f1f5f9;">{total_items}</strong> imágenes
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Grid de imágenes (3 columnas)
                cols_per_row = 3
                rows = [page_df.iloc[i:i + cols_per_row] for i in range(0, len(page_df), cols_per_row)]
                
                for row_df in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, row) in zip(cols, row_df.iterrows()):
                        with col:
                            # 🆕 Detectar si es GPT-Image para styling especial
                            herramienta = row.get('herramienta', '')
                            is_gpt_image = 'gpt-image' in str(herramienta).lower()
                            
                            estado = str(row.get('estado', 'pendiente') or 'pendiente').strip().lower()
                            estado_emoji = {
                                'pendiente': '⏳',
                                'aceptar': '✅',
                                'ajustar': '🔧',
                                'descartar': '❌'
                            }.get(estado, '❓')
                            
                            image_url = row.get("image_url", "")
                            if image_url:
                                st.image(image_url)
                            else:
                                st.markdown("""
                                <div style="height: 200px; background: #1e293b; border-radius: 8px;
                                        display: flex; align-items: center; justify-content: center; color: #64748b;">
                                    ❌ Imagen no encontrada
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Info debajo de la imagen
                            st.markdown(f"""
                            <div style="padding: 12px 0;">
                                <div style="font-weight: 600; color: #f1f5f9; font-size: 1rem; margin-bottom: 4px;">
                                    {row.get('tema', 'Sin tema').capitalize()}
                                </div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px;">
                                    📌 {row.get('concepto', 'Sin concepto')}
                                </div>
                                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                                    <span style="background: {'rgba(16, 185, 129, 0.2)' if is_gpt_image else 'rgba(168, 85, 247, 0.2)'}; 
                                                color: {'#6ee7b7' if is_gpt_image else '#c4b5fd'}; 
                                                padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;">
                                        {row.get('herramienta', 'N/A')}
                                        {'🟢' if is_gpt_image else ''}
                                    </span>
                                    <span style="background: rgba(34, 197, 94, 0.2); color: #86efac; 
                                                padding: 3px 8px; border-radius: 12px; font-size: 0.7rem;">
                                        {row.get('version', 'v1')}
                                    </span>
                                    <span style="background: rgba(251, 191, 36, 0.2); color: #fcd34d; 
                                                padding: 3px 8px; border-radius: 12px; font-size: 0.7rem;">
                                        {estado_emoji} {estado}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Botón para ver detalle
                            if st.button(f"👁️ Ver detalle", key=f"view_{row.get('image_id')}", use_container_width=True):
                                st.session_state.gallery_view_image = row.get('image_id')
                                rerun()
                
                # Paginación
                st.markdown("<br>", unsafe_allow_html=True)
                
                pag_cols = st.columns([1, 2, 1])
                
                with pag_cols[0]:
                    if st.button("◀ Anterior", disabled=(page <= 1), key="gal_prev"):
                        st.session_state.gal_page = page - 1
                        rerun()
                
                with pag_cols[1]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; color: #94a3b8;">
                        Página <strong style="color: #f1f5f9;">{page}</strong> de <strong style="color: #f1f5f9;">{total_pages}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with pag_cols[2]:
                    if st.button("Siguiente ▶", disabled=(page >= total_pages), key="gal_next"):
                        st.session_state.gal_page = page + 1
                        rerun()

# -------------------------------------------------------
# Evaluación (rúbrica 4x4) - ACTUALIZADO
# -------------------------------------------------------
elif seccion == "Evaluar (rúbrica 4x4)":

    try:
        response = supabase.table("imagenes").select("*").execute()
        df_meta = pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error al cargar imágenes: {e}")
        df_meta = pd.DataFrame()
    st.header("📊 Evaluación con rúbrica 4×4")
    
    if df_meta.empty:
        st.info("No hay imágenes en catálogo.")
    else:
        # ====== Lista de temas SIN duplicados ======
        temas_unicos = sorted(
            df_meta["tema"].dropna().astype(str).unique().tolist()
        )
        if not temas_unicos:
            st.info("No hay temas registrados en el catálogo.")
            st.stop()

        col1, col2 = st.columns([2, 1])

        # ---------------- COLUMNA IZQUIERDA: Selección y navegación ----------------
        with col1:
            # Select solo con temas únicos
            tema_sel = st.selectbox(
                "Selecciona imagen por tema",
                temas_unicos,
                key="eval_tema_sel"
            )

            # --- Estado de navegación por tema ---
            if "eval_tema_prev" not in st.session_state:
                st.session_state.eval_tema_prev = tema_sel
            if "eval_img_idx" not in st.session_state:
                st.session_state.eval_img_idx = 0

            # Si el usuario cambia de tema, reiniciar índice
            if tema_sel != st.session_state.eval_tema_prev:
                st.session_state.eval_tema_prev = tema_sel
                st.session_state.eval_img_idx = 0

            # Subconjunto de imágenes del tema elegido
            subset = df_meta[df_meta["tema"] == tema_sel].reset_index(drop=True)
            if subset.empty:
                st.warning("No hay imágenes para ese tema.")
                st.stop()

            n_imgs = len(subset)

            # Índice actual dentro del tema
            idx = st.session_state.eval_img_idx
            if idx < 0:
                idx = 0
            if idx > n_imgs - 1:
                idx = n_imgs - 1
            st.session_state.eval_img_idx = idx

            # Fila actual (imagen seleccionada)
            meta = subset.iloc[idx]

            # --- Layout: flecha izquierda · imagen · flecha derecha ---
            c_left, c_center, c_right = st.columns([1, 4, 1])

            with c_left:
                prev_btn = st.button(
                    "◀",
                    key="eval_prev_img",
                    disabled=(idx == 0)
                )

            with c_center:
                st.image(
                    meta["image_url"],
                    caption=f"{meta['tema']} / {meta['concepto']} "
                            f"[{meta['herramienta']}] (v={meta['version']})"
                )

                st.caption(f"Imagen {idx+1} de {n_imgs} para el tema «{tema_sel}»")
                
                # 🆕 MOSTRAR BADGE SI ES GPT-IMAGE 1.5
                herramienta = meta.get('herramienta', '')
                if 'gpt-image' in str(herramienta).lower():
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 6px 14px;
                        border-radius: 10px;
                        display: inline-block;
                        font-size: 0.85rem;
                        font-weight: bold;
                        margin-top: 8px;
                    ">
                        🟢 Generado con GPT-Image 1.5
                    </div>
                    """, unsafe_allow_html=True)

                # =====================================================
                # 🤖 EVALUACIÓN IA (MOSTRAR TAMBIÉN EN RÚBRICA 4×4)
                # =====================================================
                st.markdown("---")
                st.markdown("### 🤖 Evaluación IA")

                try:
                    eval_response = supabase.table("evaluaciones_ia").select("*").eq("image_id", meta["image_id"]).execute()
                    eval_imagen = pd.DataFrame(eval_response.data) if eval_response.data else pd.DataFrame()
                    
                    if not eval_imagen.empty:
                        eval_row = eval_imagen.iloc[-1]
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
                                    border-radius: 16px; padding: 20px; margin-bottom: 16px;
                                    border: 1px solid rgba(139, 92, 246, 0.2);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                <span style="font-size: 1.5rem;">✅</span>
                                <span style="color: #a78bfa; font-weight: 600; font-size: 1.1rem;">Imagen Evaluada por IA</span>
                            </div>
                            <div style="color: #64748b; font-size: 0.85rem;">
                                📅 Evaluada el: {eval_row.get('fecha', 'N/A')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        coherencia = eval_row.get('coherencia', 0)
                        fidelidad = eval_row.get('fidelidad', 0)
                        claridad = eval_row.get('claridad', 0)

                        st.markdown(f"""
                        <div style="display: flex; gap: 20px; margin: 16px 0;">
                            <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Coherencia</div>
                                <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{coherencia}/5</div>
                            </div>
                            <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Fidelidad</div>
                                <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{fidelidad}/5</div>
                            </div>
                            <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 16px; flex: 1; text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Claridad</div>
                                <div style="color: #a78bfa; font-size: 1.8rem; font-weight: bold;">{claridad}/5</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        errores = eval_row.get('errores', '')
                        if errores and str(errores).strip() and str(errores).lower() not in ['nan', 'ver conversación completa']:
                            with st.expander("⚠️ Errores Detectados", expanded=False):
                                st.markdown(f"""
                                <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 14px;
                                            border-left: 3px solid #ef4444;">
                                    <div style="color: #fca5a5; line-height: 1.6;">{errores}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        recomendaciones = eval_row.get('recomendaciones', '')
                        if recomendaciones and str(recomendaciones).strip() and str(recomendaciones).lower() != 'nan':
                            with st.expander("💡 Ver conversación completa con IA", expanded=False):
                                st.markdown(f"""
                                <div style="background: rgba(34, 197, 94, 0.1); border-radius: 10px; padding: 14px;
                                            border-left: 3px solid #22c55e; max-height: 300px; overflow-y: auto;">
                                    <div style="color: #86efac; line-height: 1.6; white-space: pre-wrap; font-size: 0.9rem;">{recomendaciones[:2000]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    else:
                        st.markdown("""
                        <div style="background: rgba(251, 191, 36, 0.1); border-radius: 12px; padding: 20px;
                                    border: 1px dashed rgba(251, 191, 36, 0.3); text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
                            <div style="color: #fcd34d; font-weight: 500;">Sin evaluación IA</div>
                            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">
                                Ve a "Evaluación IA" para analizar esta imagen
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error al cargar evaluaciones: {e}")


            with c_right:
                next_btn = st.button(
                    "▶",
                    key="eval_next_img",
                    disabled=(idx >= n_imgs - 1)
                )

            # Actualizar índice al pulsar flechas
            if prev_btn:
                st.session_state.eval_img_idx = idx - 1
                rerun()
            if next_btn:
                st.session_state.eval_img_idx = idx + 1
                rerun()

            # Info extra debajo de la imagen
            st.caption(
                f"Contraste≈{meta['wcag_ratio']} | Edge dens={meta['edge_density']} | "
                f"Alt: {str(meta['alt_text'])[:100]}..."
            )

        # ---------------- COLUMNA DERECHA: Formulario de rúbrica ----------------
        with col2:
            # Nombre del evaluador
            rater = st.text_input(
                "Nombre del evaluador (docente/estudiante)",
                value=user_name or "",
                key="eval_rater"
            )

            comentarios = {}
            puntajes = {}

            st.markdown("### Rúbrica 4×4")

            with st.form("form_rubrica"):
                # ---------- CRITERIOS UNO POR UNO ----------
                for key, q in CRITERIA:
                    st.markdown(f"**{key.capitalize()}** — {q}")

                    # Descriptores de niveles 1–4
                    descs = LEVEL_DESC.get(key, {})
                    with st.expander("Ver niveles (1–4)", expanded=False):
                        for nivel in range(1, 5):
                            txt = descs.get(nivel, "")
                            if txt:
                                st.markdown(f"- **{nivel}**: {txt}")

                    # Slider 1–4
                    puntajes[key] = st.slider(
                        f"Puntaje para {key}",
                        min_value=1,
                        max_value=4,
                        value=3,
                        step=1,
                        key=f"sl_{key}"
                    )

                    # Texto corto del nivel elegido
                    if descs.get(puntajes[key]):
                        st.caption(
                            f"Nivel seleccionado ({puntajes[key]}): "
                            f"{descs[puntajes[key]]}"
                        )

                    comentarios[key] = st.text_area(
                        f"Comentario específico sobre {key}",
                        height=60,
                        key=f"tx_{key}"
                    )

                    st.markdown("---")

                # ---------- SÍNTESIS GLOBAL ----------
                st.markdown("### Síntesis global de la imagen")

                decision_uso_key = st.radio(
                    "¿Usarías esta imagen en clase?",
                    options=list(DECISION_USO.keys()),
                    format_func=lambda k: DECISION_USO[k],
                    index=1,  # por defecto: "usar_con_ajustes"
                    key="eval_decision_uso"
                )

                severidad_key = st.radio(
                    "Severidad global de problemas detectados",
                    options=list(SEVERIDAD_GLOBAL.keys()),
                    format_func=lambda k: SEVERIDAD_GLOBAL[k],
                    index=2,  # por defecto: "moderados"
                    key="eval_severidad"
                )

                comentario_global = st.text_area(
                    "Comentario global (síntesis, ajustes recomendados)",
                    height=80,
                    key="eval_comentario_global"
                )

                enviado = st.form_submit_button(
                    "Enviar evaluación",
                    disabled=(rater.strip() == "")
                )

            # ---------- GUARDAR EN CSV ----------
            if enviado:
                now = datetime.now().isoformat(timespec="seconds")
                new_rows = []

                for key, _ in CRITERIA:
                    new_rows.append({
                        "image_id": meta["image_id"],
                        "tema": meta["tema"],
                        "concepto": meta["concepto"],
                        "herramienta": meta.get("herramienta", "N/A"),  # 🆕 Incluir herramienta
                        "rater": rater.strip() or user_name,
                        "criterio": key,
                        "puntaje": puntajes[key],
                        "comentario": comentarios[key],
                        "decision_uso": decision_uso_key,
                        "severidad": severidad_key,
                        "comentario_global": comentario_global,
                        "timestamp": now
                    })

                df_eval = pd.concat(
                    [df_eval, pd.DataFrame(new_rows)],
                    ignore_index=True
                )
                save_csv(df_eval, CSV_PATH)

                # Guardar también en Supabase
                try:
                    for new_row in new_rows:
                        supabase.table("evaluaciones_rubrica").insert(new_row).execute()
                except Exception as e:
                    st.warning(f"⚠️ No se pudo guardar en Supabase: {e}")

                log_event(
                    "rubric.submit",
                    ref_id=meta["image_id"],
                    extra={
                        "scores": puntajes,
                        "tema": meta["tema"],
                        "herramienta": meta.get("herramienta", "N/A"),  # 🆕 Log herramienta
                        "decision_uso": decision_uso_key,
                        "severidad": severidad_key
                    }
                )

                st.success("✅ Evaluación registrada correctamente.")

        # ------- Resumen de evaluaciones de ESA imagen -------
        try:
            rubrica_response = supabase.table("evaluaciones_rubrica").select("*").eq("image_id", meta["image_id"]).execute()
            df_img = pd.DataFrame(rubrica_response.data) if rubrica_response.data else pd.DataFrame()
        except Exception as e:
            st.error(f"Error cargando evaluaciones: {e}")
            df_img = pd.DataFrame()
        if df_img.empty:
            st.info("Aún no hay evaluaciones para esta imagen.")
        else:
            st.markdown("---")
            st.subheader("📊 Resumen de evaluaciones")
            
            # Promedio por criterio
            piv = df_img.pivot_table(index="criterio", values="puntaje", aggfunc="mean")
            st.write(piv)
            
            min_crit = piv["puntaje"].min()
            avg_all = piv["puntaje"].mean()
            
            sugerencia = "aceptar"
            if min_crit < 2:
                sugerencia = "descartar"
            elif min_crit < 3:
                sugerencia = "ajustar"
            
            st.success(
                f"Sugerencia: **{sugerencia}** (criterio mínimo={min_crit:.2f}, "
                f"promedio={avg_all:.2f})"
            )

            # Decisiones globales
            if {"decision_uso","severidad","comentario_global"}.issubset(df_img.columns):
                st.subheader("Decisiones globales registradas")
                resumen_global = (
                    df_img[["rater","decision_uso","severidad","comentario_global"]]
                    .drop_duplicates()
                )
                st.dataframe(resumen_global)

            # Actualizar estado (solo docente)
            if is_docente:
                nuevo_estado = st.selectbox(
                    "Actualizar estado en catálogo",
                    ["pendiente","aceptar","ajustar","descartar"],
                    index=["pendiente","aceptar","ajustar","descartar"].index(
                        meta["estado"]
                    )
                    if meta["estado"] in ["pendiente","aceptar","ajustar","descartar"]
                    else 0
                )
                if st.button("Guardar estado"):
                    df_meta.loc[
                        df_meta["image_id"] == meta["image_id"], "estado"
                    ] = nuevo_estado
                    save_csv(df_meta, META_PATH)
                    st.success("✅ Estado actualizado en el catálogo.")
                    log_event(
                        "catalog.update_state",
                        ref_id=meta["image_id"],
                        extra={"estado": nuevo_estado},
                    )
            else:
                st.info("Solo el Docente puede cambiar el estado en el catálogo.")


# -------------------------------------------------------
# 📊 RESUMEN Y REPORTES - DASHBOARD MEJORADO
# -------------------------------------------------------
elif seccion == "Resumen y reportes":

    try:
        response = supabase.table("imagenes").select("*").execute()
        df_meta = pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error al cargar imágenes: {e}")
        df_meta = pd.DataFrame()

    try:
        eval_rubrica_resp = supabase.table("evaluaciones_rubrica").select("*").execute()
        df_eval = pd.DataFrame(eval_rubrica_resp.data) if eval_rubrica_resp.data else pd.DataFrame()
    except Exception as e:
        df_eval = pd.DataFrame()

    # ========== CSS PERSONALIZADO ==========
    st.markdown("""
    <style>
    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sección de estadísticas */
    .stats-section {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Progreso bar */
    .progress-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        height: 12px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 Dashboard de Resumen y Reportes")
    
    if df_meta.empty:
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px; color: #64748b;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
            <h3 style="color: #94a3b8;">No hay datos todavía</h3>
            <p>Genera algunas imágenes para ver estadísticas</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ========== MÉTRICAS PRINCIPALES ==========
        st.markdown("### 📈 Métricas Generales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Total de imágenes
        total_imagenes = len(df_meta)
        
        # Contadores por estado
        pendientes = int((df_meta["estado"] == "pendiente").sum())
        aceptadas = int((df_meta["estado"] == "aceptar").sum())
        ajustar = int((df_meta["estado"] == "ajustar").sum())
        descartadas = int((df_meta["estado"] == "descartar").sum())
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">🖼️</div>
                <div class="metric-value">{total_imagenes}</div>
                <div class="metric-label">Total Imágenes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">⏳</div>
                <div class="metric-value">{pendientes}</div>
                <div class="metric-label">Pendientes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">✅</div>
                <div class="metric-value">{aceptadas}</div>
                <div class="metric-label">Aceptadas</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            tasa_aprobacion = (aceptadas / total_imagenes * 100) if total_imagenes > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div class="metric-value">{tasa_aprobacion:.1f}%</div>
                <div class="metric-label">Tasa Aprobación</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ========== GRÁFICOS Y ESTADÍSTICAS ==========
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown('<div class="stats-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Distribución por Estado</div>', unsafe_allow_html=True)
            
            # Crear gráfico de barras con porcentajes
            estados_data = pd.DataFrame({
                'Estado': ['Pendiente', 'Aceptar', 'Ajustar', 'Descartar'],
                'Cantidad': [pendientes, aceptadas, ajustar, descartadas],
                'Color': ['#fbbf24', '#22c55e', '#3b82f6', '#ef4444']
            })
            
            for _, row in estados_data.iterrows():
                porcentaje = (row['Cantidad'] / total_imagenes * 100) if total_imagenes > 0 else 0
                
                st.markdown(f"""
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #e2e8f0; font-weight: 500;">{row['Estado']}</span>
                        <span style="color: #94a3b8;">{row['Cantidad']} ({porcentaje:.1f}%)</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {porcentaje}%; background: {row['Color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="stats-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🛠️ Distribución por Herramienta</div>', unsafe_allow_html=True)
            
            # Contar por herramienta
            herramientas = df_meta['herramienta'].value_counts()
            
            for herramienta, cantidad in herramientas.items():
                porcentaje = (cantidad / total_imagenes * 100)
                color = '#10b981' if 'gpt-image' in str(herramienta).lower() else '#8b5cf6'
                
                st.markdown(f"""
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #e2e8f0; font-weight: 500;">{herramienta}</span>
                        <span style="color: #94a3b8;">{cantidad} ({porcentaje:.1f}%)</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {porcentaje}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== DISTRIBUCIÓN POR TEMA ==========
        st.markdown('<div class="stats-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏷️ Distribución por Tema</div>', unsafe_allow_html=True)
        
        temas = df_meta['tema'].value_counts()
        
        # Mostrar en grid de 3 columnas
        tema_cols = st.columns(min(3, len(temas)))
        
        for idx, (tema, cantidad) in enumerate(temas.items()):
            with tema_cols[idx % 3]:
                porcentaje = (cantidad / total_imagenes * 100)
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6; margin-bottom: 4px;">
                        {cantidad}
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.9rem; margin-bottom: 4px;">
                        {tema.capitalize()}
                    </div>
                    <div style="color: #64748b; font-size: 0.8rem;">
                        {porcentaje:.1f}% del total
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== MÉTRICAS TÉCNICAS PROMEDIO ==========
        st.markdown('<div class="stats-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Métricas Técnicas Promedio</div>', unsafe_allow_html=True)
        
        # Filtrar valores válidos
        wcag_values = df_meta['wcag_ratio'].dropna()
        wcag_values = wcag_values[wcag_values > 0]
        
        edge_values = df_meta['edge_density'].dropna()
        edge_values = edge_values[edge_values > 0]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            if len(wcag_values) > 0:
                wcag_avg = wcag_values.mean()
                wcag_color = "🟢" if wcag_avg >= 4.5 else ("🟡" if wcag_avg >= 3 else "🔴")
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3rem;">{wcag_color}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #3b82f6; margin: 8px 0;">
                        {wcag_avg:.2f}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Contraste WCAG Promedio
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sin datos WCAG")
        
        with col_m2:
            if len(edge_values) > 0:
                edge_avg = edge_values.mean()
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3rem;">📐</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6; margin: 8px 0;">
                        {edge_avg:.3f}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Densidad Bordes Promedio
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sin datos Edge Density")
        
        with col_m3:
            # Conteo de imágenes con evaluación IA
            try:
                eval_resp = supabase.table("evaluaciones_ia").select("image_id").execute()
                evaluadas_ia = len(set(r["image_id"] for r in eval_resp.data)) if eval_resp.data else 0
                porcentaje_ia = (evaluadas_ia / total_imagenes * 100) if total_imagenes > 0 else 0
            except:
                evaluadas_ia = 0
                porcentaje_ia = 0
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem;">🤖</div>
                <div style="font-size: 2rem; font-weight: 700; color: #10b981; margin: 8px 0;">
                    {evaluadas_ia}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    Evaluadas por IA ({porcentaje_ia:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== TABLA DE EVALUACIONES ==========
        st.markdown("---")
        st.markdown("### 📋 Promedios de Rúbrica por Imagen")
        
        if df_eval.empty:
            st.info("Aún no hay evaluaciones de rúbrica 4×4.")
        else:
            piv = df_eval.pivot_table(
                index="image_id", 
                columns="criterio", 
                values="puntaje", 
                aggfunc="mean"
            ).reset_index()
            
            merged = df_meta.merge(piv, on="image_id", how="left")
            
            # Seleccionar columnas relevantes
            cols_mostrar = ['image_id', 'tema', 'concepto', 'herramienta', 'estado']
            
            # Añadir criterios si existen
            for criterio in ['fidelidad', 'claridad', 'pertinencia', 'equidad']:
                if criterio in merged.columns:
                    cols_mostrar.append(criterio)
            
            merged_display = merged[cols_mostrar]
            
            st.dataframe(
            merged_display,
            use_container_width=True,  # ✅ ESTE SÍ EXISTE
            hide_index=True
            )
        
        # ========== EXPORTACIONES (SOLO DOCENTE) ==========
        if is_docente:
            st.markdown("---")
            st.markdown("### 📥 Exportaciones")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.markdown("""
                <div class="stats-section">
                    <div class="section-title">📥 Exportar Datos</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 16px;">
                        Descarga los datos en el formato que prefieras
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preparar datos
                merged = df_eval.merge(
                    df_meta, 
                    on=["image_id", "tema", "concepto"], 
                    how="right"
                )
                
                # Limpiar datos
                df_limpio = merged.copy()
                
                # Limpiar saltos de línea y textos
                for col in df_limpio.select_dtypes(include=['object']).columns:
                    df_limpio[col] = df_limpio[col].fillna('').apply(
                        lambda x: ' '.join(str(x).replace('\n', ' ').replace('\r', ' ').split())
                    )
                
                # Truncar prompts muy largos
                if 'prompt' in df_limpio.columns:
                    df_limpio['prompt'] = df_limpio['prompt'].apply(
                        lambda x: x[:150] + '...' if len(x) > 150 else x
                    )
                
                # Botones lado a lado
                col_csv, col_excel = st.columns(2)
                
                with col_csv:
                    # CSV
                    import io
                    csv_buffer = io.StringIO()
                    df_limpio.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                    csv_content = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📄 CSV",
                        data=csv_content,
                        file_name=f"consolidado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        
                    )
                
                with col_excel:
                    # Excel
                    excel_buffer = io.BytesIO()
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_limpio.to_excel(writer, sheet_name='Datos', index=False)
                        
                        # Formato básico
                        workbook = writer.book
                        worksheet = writer.sheets['Datos']
                        
                        # Ajustar anchos
                        for idx, col in enumerate(df_limpio.columns, 1):
                            worksheet.column_dimensions[chr(64 + idx if idx <= 26 else 64 + (idx // 26 - 1))].width = 15
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Excel",
                        data=excel_buffer,
                        file_name=f"consolidado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        
                    )
                
                st.caption(f"📊 {len(df_limpio)} registros totales")
            
            with export_col2:
                st.markdown("""
                <div class="stats-section">
                    <div class="section-title">📦 Banco Curado (ZIP)</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 16px;">
                        Descarga imágenes aceptadas + metadatos en ZIP
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Filtrar aceptadas
                aceptadas = df_meta[df_meta["estado"] == "aceptar"].copy()
                
                if aceptadas.empty:
                    st.warning("No hay imágenes aceptadas todavía.")
                else:
                    # Crear ZIP en memoria
                    import io
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
                        # Añadir CSV de metadatos
                        csv_metadata = aceptadas.to_csv(index=False, encoding="utf-8")
                        z.writestr("metadata_curado.csv", csv_metadata)
                        
                        # Añadir imágenes
                        import urllib.request
                        for _, row in aceptadas.iterrows():
                            img_url = row["image_url"]
                            img_id = row["image_id"]
                            if img_url:
                                try:
                                    with urllib.request.urlopen(img_url) as resp:
                                        img_data = resp.read()
                                    z.writestr(f"imagenes/{img_id}.png", img_data)
                                except:
                                    pass
                        
                        # Agregar README
                        readme_content = f"""# Banco Curado de Imágenes
                        
        Fecha de exportación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total de imágenes: {len(aceptadas)}
        Criterio: Imágenes con estado "aceptar"

        ## Contenido

        - metadata_curado.csv: Información de todas las imágenes
        - imagenes/: Carpeta con las imágenes aceptadas

        ## Sistema

        GEN-EDViz Monitor v1.0
        Universidad Técnica de Ambato
        """
                        z.writestr("README.txt", readme_content)
                    
                    # Preparar para descarga
                    zip_buffer.seek(0)
                    
                    # Botón de descarga directa
                    st.download_button(
                        label=f"📥 Descargar ZIP ({len(aceptadas)} imágenes)",
                        data=zip_buffer,
                        file_name=f"banco_curado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        
                    )
                    
                    st.caption(f"📦 {len(aceptadas)} imágenes incluidas")
                    
                    log_event("export.zip", extra={
                        "imagenes": len(aceptadas)
                    })
        else:
            st.info("🔒 Las exportaciones están disponibles solo para el Docente.")

# -------------------------------------------------------
# Ajustes
# -------------------------------------------------------
else:
    st.header("Ajustes y notas")
    st.write("""
- Este monitor implementa la rúbrica 4×4 colaborativa y métricas objetivas simples.
- Como apoyo a la rúbrica, el prototipo calcula métricas de contraste aproximado y densidad de bordes
  que se muestran al docente, **sin sustituir el criterio humano**.
- Incluye también un cuestionario de percepción tipo Likert y el cálculo de alfa de Cronbach para
  evaluar la consistencia interna de la escala, en línea con el perfil metodológico.
- Puedes ampliar con modelos de similitud texto–imagen (CLIP) y un pequeño predictor de calidad
  entrenado con tus propias evaluaciones (regresión a la media de la rúbrica).
- Los archivos se guardan en la carpeta ./data
""")

