"""Modèles de chat open source (interface LangChain) : un « rapide », un « fort ».

backend (settings.yaml -> llm.backend, ou variable LLM_BACKEND) :
  ollama             ChatOllama : appels d'outils + sorties structurées. Recommandé (Colab : voir notebook)
  openai_compatible  ChatOpenAI pointé vers un serveur auto-hébergé (vLLM, LM Studio, llama.cpp)
  hf                 ChatHuggingFace (transformers local) : PAS d'appel d'outils fiable ->
                     les agents reçoivent un contexte pré-calculé par le code
  mock               None : chaque nœud utilise ses règles déterministes (tests, démo instantanée)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel


def get_chat_model(kind: str, settings: dict) -> Any:
    cfg = settings["llm"]
    backend = cfg.get("backend", "mock")
    c = cfg[kind]
    if backend == "mock":
        return None
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=c["ollama"], temperature=c.get("temperature", 0.2), num_ctx=cfg.get("num_ctx", 16384),
                          base_url=os.getenv("OLLAMA_HOST", cfg.get("ollama_host", "http://localhost:11434")))
    if backend == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv(f"LLM_{kind.upper()}_MODEL", c.get("hf")),
                          base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1"),
                          api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "none"), temperature=c.get("temperature", 0.2))
    if backend == "hf":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        llm = HuggingFacePipeline.from_model_id(
            model_id=c["hf"], task="text-generation", device_map="auto",
            pipeline_kwargs={"max_new_tokens": 900, "do_sample": False, "return_full_text": False})
        return ChatHuggingFace(llm=llm)
    raise ValueError(f"backend LLM inconnu : {backend}")


def supports_tools(settings: dict) -> bool:
    """Les agents « create_agent » avec outils ne sont utilisés qu'avec ces backends."""
    return settings["llm"]["backend"] in {"ollama", "openai_compatible"}


def extract_json(text: str) -> dict | None:
    """Premier objet JSON d'un texte (tolère ```json ... ```)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def structured_call(model, schema: type[BaseModel], system: str, user: str) -> BaseModel | None:
    """Sortie structurée Pydantic : with_structured_output si possible, sinon JSON parsé."""
    msgs = [("system", system), ("human", user)]
    try:
        return model.with_structured_output(schema).invoke(msgs)
    except (NotImplementedError, AttributeError):
        pass
    except Exception as exc:  # modèle local capricieux -> on retente en JSON libre
        print(f"[llm] sortie structurée échouée ({type(exc).__name__}) -> JSON libre")
    hint = f"\nRéponds UNIQUEMENT par un JSON conforme à ce schéma :\n{json.dumps(schema.model_json_schema())}"
    try:
        raw = model.invoke([("system", system + hint), ("human", user)]).content
        data = extract_json(raw)
        return schema.model_validate(data) if data else None
    except Exception as exc:
        print(f"[llm] JSON invalide ({type(exc).__name__})")
        return None


def text_call(model, system: str, user: str) -> str:
    try:
        return model.invoke([("system", system), ("human", user)]).content.strip()
    except Exception as exc:
        print(f"[llm] appel échoué ({type(exc).__name__}: {exc})")
        return ""
