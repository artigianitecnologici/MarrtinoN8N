#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import importlib.util
from time import time
import time as time_module
from datetime import datetime
from tempfile import NamedTemporaryFile
import uuid
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import requests

# --- FLASK IMPORTS ---
from flask import (
    Flask, render_template, request, jsonify, Response, stream_with_context,
    redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from ollama import Client

# --- FASTAPI IMPORTS ---
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from faster_whisper import WhisperModel
import uvicorn

# ========= Paths & Config Comuni =========
BASE_DIR = Path(__file__).resolve().parent
BASE_PATH = str(BASE_DIR)

CONFIG_DIR = BASE_DIR / "config"
EVA_CONFIG_PATH = CONFIG_DIR / "eva_config.json"
BRIDGE_CONFIG_PATH = CONFIG_DIR / "bridge_config.json"
COMMANDS_PATH = CONFIG_DIR / "comandi.json"
BACKUP_DIR = CONFIG_DIR / "backups"

LOG_PATH = BASE_DIR / "log"
HANDLERS_PATH = BASE_DIR / "handlers"
STATE_FILE = LOG_PATH / "command_mode.state"

OUT_DIR = BASE_DIR / "out"
MODELS_DIR = BASE_DIR / "models"
WHISPER_CACHE_DIR = MODELS_DIR / "whisper_cache"

# Crea cartelle necessarie
for d in [CONFIG_DIR, LOG_PATH, HANDLERS_PATH, BACKUP_DIR, OUT_DIR, MODELS_DIR, WHISPER_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- VARIABILI AMBIENTE ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ========= Utility Logging & Dati (E.V.A.) =========
def _now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def _stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def log_error(msg: str):
    line = f"[{_now()}] [ERROR] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(os.path.join(LOG_PATH, "error.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def log_info(msg: str):
    print(f"[{_now()}] [INFO] {msg}", file=sys.stderr)

def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Impossibile leggere/parsare {path}: {e}")
        return default

def _write_json_atomic(path, data_obj):
    tmp = None
    try:
        d = json.dumps(data_obj, ensure_ascii=False, indent=2)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tf:
            tf.write(d)
            tmp = tf.name
        os.replace(tmp, path)  # atomico su POSIX
        return True
    except Exception as e:
        log_error(f"Scrittura atomica fallita per {path}: {e}")
        try:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False

def _safe_write_eva_config(new_cfg: dict) -> bool:
    try:
        current = _read_json(EVA_CONFIG_PATH, default=None)
        if current is not None:
            ts = _stamp()
            _write_json_atomic(os.path.join(BACKUP_DIR, f"eva_config.{ts}.bak.json"), current)
    except Exception as e:
        log_error(f"Backup config fallito: {e}")
    return _write_json_atomic(EVA_CONFIG_PATH, new_cfg)


# ========= Config Iniziale E.V.A. =========
EVA_CONFIG = _read_json(EVA_CONFIG_PATH, default={
    "ollama_host": "http://host.docker.internal:11434",
    "default_model": "gemma3:4b",
    "prompt_system": "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.",
    "default_profile": "default",
    "command_mode": {"enabled": True, "default_on": False},
    "profiles": {
        "default": {
            "label": "Default",
            "model": "gemma3:4b",
            "system": "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.",
            "options": {"temperature": 0.2, "top_p": 0.95, "top_k": 40, "num_ctx": 4096, "num_predict": 512, "repeat_penalty": 1.1}
        }
    }
})

def _normalize_eva_config(cfg: dict) -> dict:
    cfg = dict(cfg or {})
    cfg.setdefault("ollama_host", "http://host.docker.internal:11434")
    cfg.setdefault("default_model", "gemma3:4b")
    cfg.setdefault("prompt_system", "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.")
    cfg.setdefault("default_profile", "default")
    cfg.setdefault("command_mode", {})
    if not isinstance(cfg["command_mode"], dict): cfg["command_mode"] = {}
    cfg["command_mode"].setdefault("enabled", True)
    cfg["command_mode"].setdefault("default_on", False)
    cfg.setdefault("profiles", {})
    if not isinstance(cfg["profiles"], dict): cfg["profiles"] = {}
    if cfg["default_profile"] not in cfg["profiles"]:
        cfg["default_profile"] = "default" if "default" in cfg["profiles"] else (next(iter(cfg["profiles"]), ""))
    return cfg

EVA_CONFIG = _normalize_eva_config(EVA_CONFIG)

OLLAMA_BASE = EVA_CONFIG.get("ollama_host", "http://host.docker.internal:11434")
DEFAULT_MODEL = EVA_CONFIG.get("default_model", "gemma3:4b")
PROMPT_SYSTEM = EVA_CONFIG.get("prompt_system", "Sei E.V.A.")
DEFAULT_PROFILE = EVA_CONFIG.get("default_profile", "default")
PROFILES = EVA_CONFIG.get("profiles", {})
CMD_MODE_ENABLED = bool((EVA_CONFIG.get("command_mode") or {}).get("enabled", True))
CMD_MODE_DEFAULT_ON = bool((EVA_CONFIG.get("command_mode") or {}).get("default_on", False))

ollama_client = Client(host=OLLAMA_BASE)

# ========= GESTORE CONFIGURAZIONE BRIDGE (FastAPI) =========
DEFAULT_GLOBAL = {"ffmpeg_bin": "ffmpeg", "piper_bin": "piper", "active_profile_id": "default"}
DEFAULT_BRIDGE_PROFILE = {
    "id": "default", "name": "Standard", "whisper_size": "small", "whisper_compute": "int8",
    "piper_model_file": "it_IT-riccardo-x_low.onnx",
    "n8n_webhook_url": os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/telegram-bridge")
}

class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
        self.data = self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "global" not in data: data["global"] = DEFAULT_GLOBAL
                    if "profiles" not in data: data["profiles"] = [DEFAULT_BRIDGE_PROFILE]
                    for p in data["profiles"]:
                        if "n8n_webhook_url" not in p: p["n8n_webhook_url"] = DEFAULT_BRIDGE_PROFILE["n8n_webhook_url"]
                    return data
            except Exception:
                pass
        return {"global": DEFAULT_GLOBAL.copy(), "profiles": [DEFAULT_BRIDGE_PROFILE.copy()]}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get_global(self): return self.data["global"]
    def get_active_profile(self):
        active_id = self.data["global"].get("active_profile_id")
        for p in self.data["profiles"]:
            if p["id"] == active_id: return p
        if self.data["profiles"]:
            self.data["global"]["active_profile_id"] = self.data["profiles"][0]["id"]
            self.save()
            return self.data["profiles"][0]
        return DEFAULT_BRIDGE_PROFILE

    def get_profile_by_id(self, pid):
        for p in self.data["profiles"]:
            if p["id"] == pid: return p
        return None

    def update_profile(self, pid, new_data):
        for i, p in enumerate(self.data["profiles"]):
            if p["id"] == pid:
                self.data["profiles"][i] = {**p, **new_data}
                self.save()
                return True
        return False

    def delete_profile(self, pid):
        if len(self.data["profiles"]) <= 1: return False
        self.data["profiles"] = [p for p in self.data["profiles"] if p["id"] != pid]
        if self.data["global"]["active_profile_id"] == pid:
            self.data["global"]["active_profile_id"] = self.data["profiles"][0]["id"]
        self.save()
        return True

    def create_profile(self, name):
        new_p = DEFAULT_BRIDGE_PROFILE.copy()
        new_p["id"] = str(uuid.uuid4())[:8]
        new_p["name"] = name
        self.data["profiles"].append(new_p)
        self.save()
        return new_p["id"]

bridge_cfg = ConfigManager(BRIDGE_CONFIG_PATH)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("local-ai")

# ========= FLASK APP (E.V.A.) =========
flask_app = Flask(__name__)
flask_app.static_folder = 'static'
flask_app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
flask_app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ---- E.V.A. Handlers & Core ----
def log_to_file(question, bot_answer):
    try:
        with open(os.path.join(LOG_PATH, "log.txt"), "a", encoding="utf-8") as log_file:
            log_file.write(f"{_now()}\n[QUESTION]: {question};[OLLAMA]: {bot_answer}\n")
        if bot_answer:
            with open(os.path.join(LOG_PATH, "user.txt"), "a", encoding="utf-8") as bot_file:
                bot_file.write("user: " + str(question) + "\n")
                bot_file.write("bot: " + str(bot_answer) + "\n")
    except Exception as e:
        log_error(f"Errore scrivendo i log conversazione: {e}")

def split_string(msg):
    if isinstance(msg, str): return msg
    if isinstance(msg, dict) and "content" in msg: return str(msg["content"])
    return "(errore: contenuto non leggibile)"

def sanitize_chunk(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

def check_ollama_connectivity(raise_on_fail=False):
    try:
        r = requests.get(f"{OLLAMA_BASE.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        log_info(f"Connessione a Ollama OK su {OLLAMA_BASE}")
        return True
    except Exception as e:
        log_error(f"Impossibile connettersi a Ollama su {OLLAMA_BASE} - {e}")
        if raise_on_fail: raise
        return False

def _get_profile(name: str):
    if not name: name = EVA_CONFIG.get("default_profile") or "default"
    prof = EVA_CONFIG.get("profiles", {}).get(name)
    if prof: return name, prof
    fallback = {"label": "Compat", "model": EVA_CONFIG.get("default_model", "gemma3:4b"), "system": EVA_CONFIG.get("prompt_system", PROMPT_SYSTEM), "options": {}}
    return "compat", fallback

def _resolve_run_settings(model_from_req: str, profile_name: str):
    prof_name, prof = _get_profile(profile_name)
    model = (prof.get("model") or model_from_req or DEFAULT_MODEL).strip()
    system = (prof.get("system") or PROMPT_SYSTEM).strip()
    return model, system, prof.get("options") or {}

def get_response(messages, model_name: str, options: dict):
    try:
        response = ollama_client.chat(model=model_name, messages=messages, options=options or {})
        content = None
        try:
            msg_obj = getattr(response, "message", None)
            if msg_obj is not None: content = getattr(msg_obj, "content", None)
        except Exception: pass
        if content is None and isinstance(response, dict):
            msg_dict = response.get("message")
            if isinstance(msg_dict, dict): content = msg_dict.get("content")
        if content is None and isinstance(response, dict) and "content" in response:
            content = response["content"]
        if not isinstance(content, str) or not content.strip():
            return {"content": "(errore: formato risposta inatteso)"}
        return {"content": content}
    except Exception as e:
        return {"content": f"(errore: impossibile contattare Ollama su {OLLAMA_BASE} - {e})"}

def stream_response(messages, model_name: str, options: dict):
    def _extract_content(part):
        if isinstance(part, dict):
            msg = part.get("message") or {}
            if isinstance(msg, dict): return msg.get("content") or ""
            return part.get("response") or ""
        try:
            msg = getattr(part, "message", None)
            if msg is not None:
                if isinstance(msg, dict): return msg.get("content") or ""
                return getattr(msg, "content", "") or ""
            return getattr(part, "response", "") or ""
        except Exception: return ""

    def _generator():
        try:
            for part in ollama_client.chat(model=model_name, messages=messages, options=options or {}, stream=True):
                content = _extract_content(part)
                if content: yield sanitize_chunk(content)
        except Exception as e:
            yield f"\n[errore stream: {e}]"
    return _generator

# --- Plugin e Modalità Comandi ---
_LOADED_HANDLERS = []
def _load_handlers():
    global _LOADED_HANDLERS
    _LOADED_HANDLERS = []
    if not os.path.isdir(HANDLERS_PATH): return
    for fname in os.listdir(HANDLERS_PATH):
        if not fname.endswith(".py") or fname.startswith("_"): continue
        try:
            spec = importlib.util.spec_from_file_location(f"handlers.{fname[:-3]}", os.path.join(HANDLERS_PATH, fname))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "can_handle") and hasattr(module, "handle"):
                _LOADED_HANDLERS.append(module)
        except Exception as e:
            log_error(f"Errore caricando handler {fname}: {e}")

def try_local_handlers(text: str):
    for mod in _LOADED_HANDLERS:
        try:
            if mod.can_handle(text, {"config": EVA_CONFIG}):
                reply = mod.handle(text, {"config": EVA_CONFIG})
                if isinstance(reply, str) and reply.strip(): return sanitize_chunk(reply)
        except Exception: pass
    return None

def _read_commands():
    data = _read_json(COMMANDS_PATH, default={"prefix": "#@#", "start": [r"avvia\s+programmazione"], "stop": [r"fine\s+programmazione", r"\bstop\b"], "status": [r"\bstato\s+programmazione\b"]}) or {}
    data.setdefault("prefix", "#@#")
    return data

COMANDI = _read_commands()
CMD_PREFIX = COMANDI.get("prefix", "#@#")

def _read_command_mode() -> bool:
    try:
        if not os.path.exists(STATE_FILE): return False
        with open(STATE_FILE, "r", encoding="utf-8") as f: return f.read().strip() == "1"
    except Exception: return False

def _write_command_mode(on: bool) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f: f.write("1" if on else "0")
    except Exception: pass

def _match_any_full(patterns, text: str) -> bool:
    t = (text or "").strip()
    for p in patterns or []:
        try:
            if re.fullmatch(p, t, flags=re.IGNORECASE): return True
        except Exception: pass
    return False

def _apply_command_mode_policy(on_boot: bool = False) -> None:
    global CMD_MODE_ENABLED, CMD_MODE_DEFAULT_ON
    if not CMD_MODE_ENABLED:
        _write_command_mode(False)
        return
    if on_boot: _write_command_mode(True if CMD_MODE_DEFAULT_ON else False)

def _answer_pipeline(user_text: str, model: str, profile: str, no_handlers: bool = False, no_commands: bool = False):
    t = (user_text or "").strip()
    if (not no_commands) and CMD_MODE_ENABLED:
        if _match_any_full(COMANDI.get("start"), t):
            _write_command_mode(True)
            return "Modalita comandi ATTIVATA"
        if _match_any_full(COMANDI.get("stop"), t):
            _write_command_mode(False)
            return "Modalita comandi DISATTIVATA. Torno a usare il modello."
        if _match_any_full(COMANDI.get("status"), t):
            return "Modalita comandi: ON" if _read_command_mode() else "Modalita comandi: OFF"
        if _read_command_mode():
            return f"{CMD_PREFIX}{t if t else '(vuoto)'}"
    else:
        if not CMD_MODE_ENABLED and _read_command_mode(): _write_command_mode(False)

    if not no_handlers:
        local = try_local_handlers(t)
        if local is not None: return local

    model_res, system_prompt, options = _resolve_run_settings(model, profile)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": t}]
    new_msg = get_response(messages, model_res, options)
    return sanitize_chunk(split_string(new_msg.get('content', new_msg)))

# --- FLASK ROUTES ---
@flask_app.route("/")
def home():
    model_names = [DEFAULT_MODEL]
    try:
        r = requests.get(f"{OLLAMA_BASE.rstrip('/')}/api/tags", timeout=5)
        if r.status_code == 200:
            model_names = [(m.get("name") or m.get("model")) for m in r.json().get("models", []) if (m.get("name") or m.get("model"))] or [DEFAULT_MODEL]
    except Exception: pass
    return render_template("indexollama.html", models=model_names, default_model=DEFAULT_MODEL, profiles=EVA_CONFIG.get("profiles", {}), default_profile=EVA_CONFIG.get("default_profile", "default"))

@flask_app.route("/get")
def get_bot_response():
    q, model, profile = request.args.get('msg', ''), request.args.get('model', DEFAULT_MODEL), request.args.get('profile', EVA_CONFIG.get("default_profile", "default"))
    msgout = _answer_pipeline(q, model, profile)
    log_to_file(q, msgout)
    return msgout

# @flask_app.route('/json', methods=['GET', 'POST'])
# def json_response():
#     data = request.get_json(silent=True) or {} if request.method == 'POST' else request.args
#     q = (data.get('query') or '').strip()
#     msgout = _answer_pipeline(q, data.get('model', DEFAULT_MODEL), data.get('profile', EVA_CONFIG.get("default_profile", "default")))
#     return jsonify({"response": msgout, "action": "ok"})

@flask_app.route('/json', methods=['GET', 'POST'])
def json_response():
    try:
        data = request.get_json(silent=True) if request.method == 'POST' else request.args
        data = data or {}

        q = (data.get('query') or data.get('text') or data.get('message') or '').strip()
        if not q:
            return jsonify({
                "response": "",
                "action": "error",
                "error": "Parametro query mancante"
            }), 400

        profile = bridge_cfg.get_active_profile()
        n8n_webhook = profile.get(
            "n8n_webhook_url",
            "http://n8n:5678/webhook/telegram-bridge"
        )

        session_id = (
            data.get("sessionId")
            or data.get("session_id")
            or request.remote_addr
            or f"web_{uuid.uuid4().hex[:8]}"
        )

        chat_id = data.get("chat_id")

        msgout = _blocking_n8n_webhook(q, n8n_webhook, session_id, chat_id)

        log_to_file(q, msgout)

        return jsonify({
            "response": msgout,
            "action": "ok",
            "sessionId": session_id
        })

    except Exception as e:
        log_error(f"Errore endpoint /json: {e}")
        return jsonify({
            "response": f"Errore interno: {e}",
            "action": "error"
        }), 500
# ========= FASTAPI APP (BRIDGE N8N) =========
fastapi_app = FastAPI(title="Local-AI Server Pro & EVA Bridge")
executor = ThreadPoolExecutor(max_workers=3)

global_history: List[dict] = []
current_whisper_model = None
loaded_whisper_size = None

def load_whisper_blocking(size, compute):
    global current_whisper_model, loaded_whisper_size
    if current_whisper_model and loaded_whisper_size == size: return current_whisper_model
    log.info(f"--- ⏳ CARICAMENTO WHISPER: {size} ---")
    try:
        model = WhisperModel(size, device="cpu", compute_type=compute, download_root=str(WHISPER_CACHE_DIR))
        current_whisper_model = model
        loaded_whisper_size = size
        log.info(f"--- ✅ WHISPER {size} PRONTO ---")
        return model
    except Exception as e:
        log.error(f"Errore critico caricamento Whisper: {e}")
        return None

# --- Telegram Background Task ---
async def handle_telegram_message(msg):
    chat_id = msg["chat"]["id"]
    user_content = None
    loop = asyncio.get_running_loop()

    if "text" in msg:
        user_content = msg["text"]
    elif "voice" in msg:
        file_id = msg["voice"]["file_id"]
        def _download_tg_audio():
            res = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
            if not res.get("result", {}).get("file_path"): return None
            return requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{res['result']['file_path']}").content
            
        audio_data = await loop.run_in_executor(executor, _download_tg_audio)
        if audio_data:
            temp_ogg = OUT_DIR / f"tg_{uuid.uuid4()}.ogg"
            with open(temp_ogg, "wb") as f: f.write(audio_data)
            
            def _transcribe_tg():
                if not current_whisper_model: return None
                segments, _ = current_whisper_model.transcribe(str(temp_ogg), language="it", beam_size=5)
                return " ".join([s.text for s in segments]).strip()

            user_content = await loop.run_in_executor(executor, _transcribe_tg)
            try: os.remove(temp_ogg)
            except: pass

    if user_content:
        profile = bridge_cfg.get_active_profile()
        n8n_webhook = profile.get("n8n_webhook_url", "http://n8n:5678/webhook/telegram-bridge")
        reply_text = await loop.run_in_executor(executor, _blocking_n8n_webhook, user_content, n8n_webhook, f"tg_{chat_id}", chat_id)
        await loop.run_in_executor(executor, lambda: requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": reply_text}))

async def telegram_bridge_task():
    if not TELEGRAM_BOT_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN mancante. Bridge Telegram disabilitato.")
        return
    last_update_id = None
    loop = asyncio.get_running_loop()
    while True:
        try:
            resp = await loop.run_in_executor(executor, lambda: requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates", params={"offset": last_update_id, "timeout": 10}).json())
            if resp and "result" in resp:
                for update in resp["result"]:
                    last_update_id = update["update_id"] + 1
                    if "message" in update: asyncio.create_task(handle_telegram_message(update["message"]))
        except Exception:
            await asyncio.sleep(5)
        await asyncio.sleep(1)

@fastapi_app.on_event("startup")
async def startup_event():
    p = bridge_cfg.get_active_profile()
    load_whisper_blocking(p.get("whisper_size", "small"), p.get("whisper_compute", "int8"))
    asyncio.create_task(telegram_bridge_task())

# --- Utilities Audio ---
def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[^\x00-\x7F\u00C0-\u017F\s.,!?\'"-]+', '', text).replace("**", "").replace("*", "").replace("#", "")
    return re.sub(r'\s+', ' ', re.sub(r'[\(\*][^*)]+[\)\*]', '', text)).strip()

# def _docker_url_fix(url: str, service_name: str) -> str:
#     return url.replace("localhost", service_name).replace("127.0.0.1", service_name) if "localhost" in url or "127.0.0.1" in url else url
def _docker_url_fix(url: str, service_name: str) -> str:
    return url

def _blocking_decode(audio_bytes, ffmpeg_bin):
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"]
    return subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

def _blocking_stt_auto(pcm16):
    if not current_whisper_model: return ""
    segments, _ = current_whisper_model.transcribe((np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0), language="it", vad_filter=True)
    return "".join(s.text for s in segments).strip()

# def _blocking_n8n_webhook(text, webhook_url, session_id, chat_id=None):
#     try:
#         r = requests.post(_docker_url_fix(webhook_url, "n8n"), json={"text": text, "message": text, "sessionId": session_id, "chat_id": chat_id}, timeout=60)
#         if r.status_code == 200:
#             resp = r.json()
#             if isinstance(resp, list) and len(resp) > 0: return resp[0].get("output", resp[0].get("text", str(resp[0])))
#             elif isinstance(resp, dict): return resp.get("output", resp.get("text", str(resp)))
#             return str(resp)
#         return f"Errore server n8n: {r.status_code}"
#     except Exception as e: return f"Errore connessione a n8n: {e}"
def _blocking_n8n_webhook(text, webhook_url, session_id, chat_id=None):
    try:
        payload = {
            "text": text,
            "message": text,
            "sessionId": session_id,
            "chat_id": chat_id
        }

        log_info(f"Webhook n8n usato: {webhook_url}")

        r = requests.post(webhook_url, json=payload, timeout=60)
        r.raise_for_status()

        try:
            resp = r.json()
        except Exception:
            return r.text.strip() if r.text else "Webhook n8n ha risposto senza JSON"

        if isinstance(resp, list) and len(resp) > 0:
            first = resp[0]
            if isinstance(first, dict):
                return first.get("output", first.get("text", str(first)))
            return str(first)

        if isinstance(resp, dict):
            return resp.get("output", resp.get("text", str(resp)))

        return str(resp)

    except requests.exceptions.RequestException as e:
        return f"Errore connessione a n8n: {e}"
    except Exception as e:
        return f"Errore generico n8n: {e}"
        
def _blocking_tts(text, wav_path, profile, piper_bin):
    subprocess.run([piper_bin, "--model", str(MODELS_DIR / profile.get("piper_model_file", "it_IT-riccardo-x_low.onnx")), "--output_file", str(wav_path)], input=text.encode("utf-8"), check=True)

async def process_audio_pipeline(audio_bytes: bytes, session_id: str):
    loop = asyncio.get_running_loop()
    glob_cfg, profile = bridge_cfg.get_global(), bridge_cfg.get_active_profile()
    if not current_whisper_model: return None, None, "Whisper non caricato."

    try:
        transcript = await loop.run_in_executor(executor, _blocking_stt_auto, await loop.run_in_executor(executor, _blocking_decode, audio_bytes, glob_cfg["ffmpeg_bin"]))
    except Exception as e: return None, None, f"Err Input: {e}"

    if not transcript: return None, None, "Silenzio"
    global_history.append({"role": "user", "content": transcript, "time": time_module.strftime("%H:%M:%S")})

    raw_reply = await loop.run_in_executor(executor, _blocking_n8n_webhook, transcript, profile.get("n8n_webhook_url", "http://n8n:5678/webhook/telegram-bridge"), session_id, None)
    clean_reply = clean_text_for_tts(raw_reply) or "Scusa, non ho una risposta valida."
    global_history.append({"role": "assistant", "content": clean_reply, "time": time_module.strftime("%H:%M:%S")})

    audio_filename = f"{uuid.uuid4()}.wav"
    try:
        await loop.run_in_executor(executor, _blocking_tts, clean_reply, OUT_DIR / audio_filename, profile, glob_cfg["piper_bin"])
    except Exception as e: return transcript, clean_reply, f"Err TTS: {e}"
    return transcript, clean_reply, f"/api/audio/{audio_filename}"

# --- FASTAPI ROUTES ---
@fastapi_app.get("/api/history")
def get_history():
    return JSONResponse(global_history[-20:])

@fastapi_app.post("/api/voice_raw")
async def voice_raw(req: Request):
    if req.query_params.get("reset") == "true": return {"status": "memory_cleared"}
    body = await req.body()
    if not body: return JSONResponse({"error": "No audio"}, status_code=400)
    transcript, reply, audio_url = await process_audio_pipeline(body, req.client.host if req.client else "unknown")
    if not reply: return JSONResponse({"error": audio_url, "transcript": transcript}, status_code=500)
    return {"transcript": transcript, "reply": reply, "audio_url": audio_url}

@fastapi_app.get("/api/audio/{name}")
async def get_audio(name: str):
    path = OUT_DIR / name
    if not path.exists(): raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


# ========= AVVIO COMBINATO =========
def run_flask_app():
    # Disattiviamo il reloader per non mandare in crash il thread
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    log_info(f"Avvio combinato E.V.A. (Flask) e Bridge (FastAPI)...")
    _load_handlers()
    check_ollama_connectivity(False)
    _apply_command_mode_policy(on_boot=True)

    # 1. Lancia Flask in un thread demone sulla porta 5000
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    log_info("Flask E.V.A. in esecuzione su http://0.0.0.0:5000")

    # 2. Lancia FastAPI nel main thread sulla porta 8000
    log_info("FastAPI Bridge in esecuzione su http://0.0.0.0:8000")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)