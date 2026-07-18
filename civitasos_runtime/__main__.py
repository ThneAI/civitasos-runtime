"""CLI entry point — ``python -m civitasos_runtime``.

Usage examples::

    # Minimal — Ollama local, default node
    python -m civitasos_runtime --name AlphaTrader --llm ollama:qwen3

    # Full options
    python -m civitasos_runtime \
        --name BetaScout \
        --backend http://node1:8099 \
        --llm openai:gpt-4o --llm-api-key sk-... \
        --capabilities trading,scouting \
        --gateway-port 8300 \
        --identity agent-beta.key \
        --stake 200

    # Multiple backend nodes (failover)
    python -m civitasos_runtime \
        --name GammaWorker \
        --backend http://node1:8099,http://node2:8099 \
        --llm anthropic:claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="civitasos-runtime",
        description="CivitasOS Agent Runtime — run an autonomous agent.",
    )

    # Required
    p.add_argument(
        "--name", "-n",
        default=os.getenv("AGENT_NAME", "CivitasAgent"),
        help="Agent display name (env: AGENT_NAME)",
    )

    # Backend
    p.add_argument(
        "--backend", "-b",
        default=os.getenv("CIVITASOS_URL", "http://localhost:8099"),
        help="CivitasOS backend URL(s), comma-separated for failover (env: CIVITASOS_URL)",
    )

    # LLM
    p.add_argument(
        "--llm",
        default=os.getenv("AGENT_LLM", "openai:qwen3:latest"),
        help=(
            "LLM spec: provider:model. "
            "Examples: ollama:qwen3, openai:gpt-4o, anthropic:claude-sonnet-4-20250514, litellm:... "
            "(env: AGENT_LLM)"
        ),
    )
    p.add_argument(
        "--llm-api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="API key for the LLM provider (env: LLM_API_KEY)",
    )
    p.add_argument(
        "--llm-base-url",
        default=os.getenv("LLM_BASE_URL", ""),
        help="Override LLM base URL (env: LLM_BASE_URL)",
    )

    # Capabilities
    p.add_argument(
        "--capabilities", "-c",
        default=os.getenv("AGENT_CAPABILITIES", "general"),
        help="Comma-separated capabilities (env: AGENT_CAPABILITIES)",
    )

    # Gateway
    p.add_argument(
        "--gateway-port", "-g",
        type=int,
        default=int(os.getenv("GATEWAY_PORT", "0")) or None,
        help="Enable HTTP gateway on this port (env: GATEWAY_PORT)",
    )

    # Identity
    p.add_argument(
        "--identity", "-i",
        default=os.getenv("AGENT_IDENTITY", ""),
        help="Path to identity key file for persistent DID (env: AGENT_IDENTITY)",
    )

    # Economics
    p.add_argument(
        "--stake",
        type=int,
        default=int(os.getenv("AGENT_STAKE", "100")),
        help="Initial stake amount in CIV (env: AGENT_STAKE)",
    )
    p.add_argument(
        "--heartbeat",
        type=int,
        default=int(os.getenv("AGENT_HEARTBEAT", "60")),
        help="Heartbeat interval in seconds (env: AGENT_HEARTBEAT)",
    )

    # Debug
    p.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--endpoint",
        default=os.getenv("AGENT_ENDPOINT", ""),
        help="Public URL where this agent's gateway is reachable by other agents (env: AGENT_ENDPOINT)",
    )

    p.add_argument(
        "--data-dir",
        default=os.getenv("AGENT_DATA_DIR", "data"),
        help="Directory for persistent local memory (env: AGENT_DATA_DIR)",
    )

    p.add_argument(
        "--checkpoint-root",
        default=os.getenv("CIVITASOS_CHECKPOINT_ROOT", ""),
        help="Atomic identity checkpoint root (env: CIVITASOS_CHECKPOINT_ROOT)",
    )
    p.add_argument(
        "--restore-checkpoint-on-start",
        action="store_true",
        default=_env_true("CIVITASOS_RESTORE_CHECKPOINT_ON_START"),
        help="Restore the active checkpoint before gateway, heartbeat, or ticks",
    )

    return p


def _resolve_ollama(spec: str, base_url: str) -> tuple[str, dict]:
    """Handle 'ollama:model' shorthand → OpenAI adapter with Ollama base_url."""
    if spec.startswith("ollama:"):
        model = spec.split(":", 1)[1]
        url = base_url or "http://localhost:11434/v1"
        return f"openai:{model}", {"api_key": "ollama", "base_url": url}
    return spec, {}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Parse multi-node backend
    backends = [u.strip() for u in args.backend.split(",") if u.strip()]
    base_url: str | list[str] = backends[0] if len(backends) == 1 else backends

    # Resolve LLM
    llm_spec, extra_kwargs = _resolve_ollama(args.llm, args.llm_base_url)
    if args.llm_api_key and "api_key" not in extra_kwargs:
        extra_kwargs["api_key"] = args.llm_api_key
    if args.llm_base_url and "base_url" not in extra_kwargs:
        extra_kwargs["base_url"] = args.llm_base_url

    from .runner import AgentRunner

    runner = AgentRunner(
        base_url=base_url,
        name=args.name,
        capabilities=[c.strip() for c in args.capabilities.split(",") if c.strip()],
        llm=llm_spec,
        llm_kwargs=extra_kwargs or None,
        stake=args.stake,
        heartbeat_interval=args.heartbeat,
        gateway_port=args.gateway_port,
        identity_file=args.identity or None,
        endpoint_url=args.endpoint or None,
        data_dir=args.data_dir,
        checkpoint_root=args.checkpoint_root or None,
        restore_checkpoint_on_start=args.restore_checkpoint_on_start,
    )

    logger = logging.getLogger("civitasos_runtime")
    logger.info(
        "Starting %s | backend=%s | llm=%s | gateway=%s",
        args.name,
        args.backend,
        args.llm,
        args.gateway_port or "off",
    )

    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


if __name__ == "__main__":
    main()
