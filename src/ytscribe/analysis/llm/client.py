"""Router construction: free-tier cloud first, local quantized Ollama as fallback (llmrouter-free). A deployment whose key is
not set is skipped. The local deployment only needs OLLAMA_API_KEY to be non-empty (it is not a secret), so it defaults to
'ollama'; cloud keys are never invented."""

import os
from pathlib import Path

from llmrouter_free import build_router, load_routes
from loguru import logger
from sqlalchemy import create_engine

from ytscribe.settings import get_settings, load_env

LLM_KEY_NAMES = ("OPENROUTER_API_KEY",)  # the only variables ever read from llm_env_file; values are never printed or logged


def load_llm_keys(env_file: str) -> list[str]:
    """Export just the named provider keys from `env_file` (if it exists) into this process. Returns the NAMES loaded."""
    from dotenv import dotenv_values

    path = Path(env_file).expanduser()
    if not env_file or not path.is_file():
        return []
    vals = dotenv_values(path)
    loaded = []
    for name in LLM_KEY_NAMES:
        if vals.get(name) and not os.environ.get(name):
            os.environ[name] = vals[name]
            loaded.append(name)
    return loaded


def get_router(routes_file: str = "llm_routes.yaml"):
    load_env()
    s = get_settings()
    loaded = load_llm_keys(s.llm_env_file)
    if loaded:
        logger.info("provider keys {} loaded from {}", loaded, Path(s.llm_env_file).expanduser())
    os.environ.setdefault("OLLAMA_API_KEY", "ollama")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{Path(s.data_dir) / 'llm.db'}")
    return build_router(load_routes(s.configs_dir / routes_file), engine=engine)


def enabled_deployments(router) -> list[str]:
    return [name for name, d in router.deployments.items() if d.enabled]
