"""ToolRegistry — auto-discover SDK methods and expose them as LLM function schemas.

Introspects CivitasAgent mixin methods and generates OpenAI-compatible tool
definitions that the LLM can invoke via function calling.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

from .models import ToolDef

logger = logging.getLogger(__name__)

# Methods to skip when auto-discovering SDK tools
_SKIP_METHODS = frozenset({
    # Internal / lifecycle
    "generate_keys", "load_keys", "sign", "authenticate",
    "save_identity", "load_identity", "set_api_version",
    "discover_nodes", "wait_ready", "ping",
    # Event wakeup is managed by AgentRunner. Exposing these to the LLM causes
    # malformed event subscriptions and does not help task execution.
    "webhook_register", "webhook_unregister",
    # Legacy SDK facade helpers depend on delegate_task, which is not available
    # on the current CivitasAgent surface. Keeping them exposed makes external
    # models pick dead-end tools instead of pool/read/execute actions.
    "ask_guardian", "find_best_agent", "negotiate_chain",
    "post_to_marketplace", "query_reputation",
    "refresh_token", "list_system_agents",
    # Worker pattern (we ARE the loop)
    "start_worker", "stop_worker", "task_handler",
    # CSP management (handled at runner level)
    "csp_discover", "csp_bind", "csp_unbind",
})

# Category rules: action prefix → (category, requires_conscience, est_cost)
_CATEGORY_RULES: list[tuple[str, str, bool, float]] = [
    # (prefix, category, requires_conscience, estimated_cost)
    ("briefing", "perceive", False, 0),
    ("pool_discover", "perceive", False, 0),
    ("a2a_discover", "perceive", False, 0),
    ("a2a_list", "perceive", False, 0),
    ("recall", "memory", False, 0),
    ("remember", "memory", False, 0.1),
    ("forget", "memory", False, 0.1),
    ("log_episode", "memory", False, 0.2),
    ("replay_episodes", "memory", False, 0),
    ("memory_", "memory", False, 0),
    ("pool_claim", "task", True, 1.0),
    ("pool_post", "task", True, 1.0),
    ("pool_complete", "task", True, 0.5),
    ("pool_fail", "task", True, 0.5),
    ("task_execute", "task", True, 2.0),
    ("task_settle", "task", True, 1.0),
    ("economics_", "economic", True, 1.0),
    ("r2r_", "social", True, 0.5),
    ("create_proposal", "governance", True, 5.0),
    ("vote", "governance", True, 2.0),
    ("get_", "info", False, 0),
    ("list_", "info", False, 0),
    ("health", "system", False, 0),
    ("heartbeat", "system", False, 0),
    ("get_status", "system", False, 0),
]


def _classify(name: str) -> tuple[str, bool, float]:
    """Classify a method name into (category, requires_conscience, cost)."""
    for prefix, cat, conscience, cost in _CATEGORY_RULES:
        if name.startswith(prefix):
            return cat, conscience, cost
    return "general", False, 1.0


def _annotation_text(annotation: Any) -> str:
    if annotation == inspect.Parameter.empty:
        return ""
    return str(annotation)


def _annotation_json_type(annotation: Any) -> str:
    type_name = _annotation_text(annotation)
    if not type_name:
        return "string"

    lower = type_name.lower()
    # Container types must be detected before scalar item types. For example
    # Optional[List[str]] contains "str", but the function schema must be array.
    if any(
        marker in lower
        for marker in ("list[", "list,", "typing.list", "sequence[", "tuple[", "set[")
    ):
        return "array"
    if any(
        marker in lower
        for marker in ("dict[", "dict,", "typing.dict", "mapping[")
    ):
        return "object"
    if "bool" in lower:
        return "boolean"
    if "int" in lower:
        return "integer"
    if "float" in lower:
        return "number"
    if "str" in lower:
        return "string"
    return "string"


def _param_schema(param: inspect.Parameter) -> dict[str, Any]:
    """Convert a single inspect.Parameter to a JSON Schema property."""
    schema: dict[str, Any] = {"type": _annotation_json_type(param.annotation)}

    if param.default not in (inspect.Parameter.empty, None):
        schema["default"] = param.default

    return schema


def _param_expects_array(name: str, param: inspect.Parameter) -> bool:
    if name == "capabilities":
        return True
    return _annotation_json_type(param.annotation) == "array"


def _normalize_param_value(name: str, param: inspect.Parameter, value: Any) -> Any:
    if _param_expects_array(name, param):
        return _normalize_array_value(value)
    return value


def _normalize_array_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if not isinstance(value, str):
        return [value]

    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
        if parsed not in (None, ""):
            return [parsed]
    return [item.strip() for item in text.split(",") if item.strip()]


class ToolRegistry:
    """Discovers SDK methods and custom tools, converts to LLM tool schemas."""

    def __init__(self, agent: Any = None) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._executors: dict[str, Callable[..., Any]] = {}
        if agent is not None:
            self.discover(agent)

    # -- Auto-discovery from SDK --------------------------------------------

    def discover(self, agent: Any) -> int:
        """Introspect *agent* (CivitasAgent instance) and register public methods."""
        count = 0
        for name in dir(agent):
            if name.startswith("_"):
                continue
            if name in _SKIP_METHODS:
                continue
            attr = getattr(agent, name, None)
            if not callable(attr):
                continue

            sig = inspect.signature(attr)
            cat, conscience, cost = _classify(name)

            # Build JSON Schema for parameters
            properties: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                properties[pname] = _param_schema(param)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)

            params_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                params_schema["required"] = required

            doc = inspect.getdoc(attr) or ""
            description = doc.split("\n")[0] if doc else name.replace("_", " ").title()

            tool = ToolDef(
                name=name,
                description=description,
                parameters=params_schema,
                category=cat,
                requires_conscience=conscience,
                estimated_cost=cost,
            )
            self._tools[name] = tool
            self._executors[name] = attr
            count += 1
            logger.debug("Discovered tool: %s [%s]", name, cat)

        logger.info("ToolRegistry: discovered %d tools from SDK", count)
        return count

    # -- Manual tool registration -------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        requires_conscience: bool = False,
        estimated_cost: float = 0.0,
    ) -> None:
        """Manually register a custom tool."""
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            properties[pname] = _param_schema(param)
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        params_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            params_schema["required"] = required

        tool = ToolDef(
            name=name,
            description=description or name,
            parameters=params_schema,
            category="custom",
            requires_conscience=requires_conscience,
            estimated_cost=estimated_cost,
        )
        self._tools[name] = tool
        self._executors[name] = fn

    # -- Tool schema export -------------------------------------------------

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Export all tools in OpenAI function calling format."""
        result = []
        for tool in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    # -- Execution ----------------------------------------------------------

    @staticmethod
    def _filter_params(fn: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs the target callable does not accept (LLM hallucination guard)."""
        if not params:
            return {}
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return dict(params)
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kw:
            return dict(params)
        accepted = {
            n: p for n, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
            and n != "self"
        }
        cleaned: dict[str, Any] = {}
        dropped: list[str] = []
        fn_name = getattr(fn, "__name__", "")
        if fn_name == "task_execute" and "output" in accepted and "output" not in params:
            for alias in ("result", "content", "response", "answer"):
                if params.get(alias) not in (None, ""):
                    params = dict(params)
                    params["output"] = params[alias]
                    break
        for k, v in params.items():
            if k in accepted:
                cleaned[k] = _normalize_param_value(k, accepted[k], v)
            elif k.startswith("_"):
                # Internal control metadata used by benchmark/rules path.
                # Never forward to SDK calls and do not warn as hallucination.
                continue
            else:
                dropped.append(k)

        # Normalize known SDK schema hotspots so malformed LLM args do not
        # hard-fail the HTTP layer (e.g., metadata must be Dict[str, str]).
        if fn_name == "task_execute" and "metadata" in cleaned:
            metadata = cleaned.get("metadata")
            if isinstance(metadata, dict):
                cleaned["metadata"] = {
                    str(k): (v if isinstance(v, str) else str(v))
                    for k, v in metadata.items()
                }
            else:
                cleaned.pop("metadata", None)

        if dropped:
            logger.warning(
                "Tool '%s': dropping unknown kwargs from LLM: %s",
                getattr(fn, "__name__", "<fn>"), dropped,
            )
        return cleaned

    def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given params."""
        fn = self._executors.get(name)
        if fn is None:
            raise KeyError(f"Tool '{name}' not registered.")
        return fn(**self._filter_params(fn, params))

    async def aexecute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool, awaiting if it's async."""
        fn = self._executors.get(name)
        if fn is None:
            raise KeyError(f"Tool '{name}' not registered.")
        result = fn(**self._filter_params(fn, params))
        if inspect.isawaitable(result):
            return await result
        return result

    # -- Accessors ----------------------------------------------------------

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.category == category]

    @property
    def names(self) -> set[str]:
        return set(self._tools.keys())
