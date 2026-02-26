"""Hermes Financial Research — Gradio chat demo.

Run with environment variables for your LLM provider:

    # Anthropic (default)
    ANTHROPIC_API_KEY=sk-ant-... \\
    HERMES_SEC_USER_AGENT="MyApp/1.0 me@company.com" \\
    python demo/app.py

    # OpenAI
    HERMES_LLM_PROVIDER=openai HERMES_LLM_MODEL=gpt-4o \\
    OPENAI_API_KEY=sk-... \\
    HERMES_SEC_USER_AGENT="MyApp/1.0 me@company.com" \\
    python demo/app.py

Optional:
    HERMES_FRED_API_KEY=...   — enables FRED macroeconomic tools
    PORT=7860                 — override the default port
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncGenerator
from pathlib import Path

import gradio as gr

import hermes
from hermes.config import HermesConfig
from hermes.infra.streaming import EventType

# ---------------------------------------------------------------------------
# Rate limiting — one concurrent request per IP
# ---------------------------------------------------------------------------

_active_ips: set[str] = set()
_ip_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Tool display helpers
# ---------------------------------------------------------------------------

_TOOL_ICONS: dict[str, str] = {
    # SEC EDGAR
    "get_company_facts": "📊",
    "search_filings": "🔍",
    "get_submissions": "📋",
    "get_filing_urls": "🔗",
    "get_filing_financial_tables": "📈",
    "get_filing_text": "📄",
    "get_filing_content": "📑",
    "get_insider_transactions": "👤",
    "get_institutional_holdings": "🏦",
    # FRED
    "get_series": "📉",
    "search_series": "🔎",
    "get_series_info": "ℹ️",
    # Market data
    "get_quote": "💹",
    "get_history": "📅",
    "get_batch_quotes": "📊",
    # News
    "get_news": "📰",
    "get_financial_news": "🗞️",
    # Excel
    "excel_create_workbook": "📗",
    "excel_write_cells": "✏️",
    "excel_read_range": "👁️",
    "excel_add_formula": "🔢",
    "excel_format_range": "🎨",
    "excel_add_chart": "📊",
    "excel_add_sheet": "➕",
    "excel_audit_workbook": "🔍",
    "excel_save": "💾",
    # Documents
    "document_create": "📝",
    "document_add_heading": "📌",
    "document_add_paragraph": "✍️",
    "document_add_table": "📋",
    "document_add_image": "🖼️",
    "document_save": "💾",
    # Charts
    "create_line_chart": "📈",
    "create_bar_chart": "📊",
    "create_waterfall_chart": "🌊",
    "create_scatter_chart": "🔵",
    "create_heatmap": "🌡️",
}

_FILE_EXTENSIONS = {".xlsx", ".docx", ".pdf", ".png"}
_DEFAULT_ICON = "🔧"


def _tool_line(name: str, status: str) -> str:
    icon = _TOOL_ICONS.get(name, _DEFAULT_ICON)
    badge = {"running": "⏳", "success": "✅", "error": "❌"}.get(status, "⏳")
    return f"{icon} `{name}` {badge}"


def _build_message(
    tool_calls: list[list[str]],
    response_text: str,
    generated_files: list[str],
) -> str:
    parts: list[str] = []
    if tool_calls:
        parts.append("\n".join(_tool_line(n, s) for n, s in tool_calls))
    if response_text:
        parts.append(response_text)
    if generated_files:
        names = "\n".join(f"📎 **{Path(f).name}**" for f in generated_files)
        parts.append(f"---\n**Files generated** (download below):\n{names}")
    return "\n\n".join(parts) if parts else "⏳ Working..."


# ---------------------------------------------------------------------------
# Streaming response handler
# ---------------------------------------------------------------------------


async def respond(
    message: str,
    history: list[dict],
    provider: str,
    model: str,
    sec_user_agent: str,
    fred_api_key: str,
    request: gr.Request,
) -> AsyncGenerator[tuple[list[dict], list[str] | None], None]:
    """Drive a Hermes research query and stream updates to the chatbot."""
    ip: str = getattr(getattr(request, "client", None), "host", "unknown")

    # One concurrent request per IP
    with _ip_lock:
        if ip in _active_ips:
            yield history + [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ A request from your IP is already in progress. "
                        "Please wait for it to finish before submitting another."
                    ),
                },
            ], None
            return
        _active_ips.add(ip)

    try:
        agent_str = sec_user_agent.strip() or os.environ.get("HERMES_SEC_USER_AGENT", "")
        if not agent_str:
            yield history + [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ **SEC User Agent is required.**\n\n"
                        "Open ⚙️ Settings above and enter a value like "
                        "`YourApp/1.0 you@email.com`. The SEC requires this "
                        "to identify your application."
                    ),
                },
            ], None
            return

        cfg = HermesConfig(
            llm_provider=provider,
            llm_model=model.strip(),
            sec_user_agent=agent_str,
            fred_api_key=fred_api_key.strip() or None,
        )

        # Append user turn; open empty assistant bubble immediately
        history = history + [{"role": "user", "content": message}]
        history = history + [{"role": "assistant", "content": "⏳ Working..."}]
        yield history, None

        # Streaming state
        tool_calls: list[list[str]] = []
        last_running_idx: dict[str, int] = {}  # tool_name → index of last ⏳ entry
        response_text = ""
        generated_files: list[str] = []

        h = hermes.Hermes(config=cfg)

        async for event in h.stream(message):
            if event.type == EventType.TOOL_CALL and event.tool_name:
                idx = len(tool_calls)
                tool_calls.append([event.tool_name, "running"])
                last_running_idx[event.tool_name] = idx

            elif event.type == EventType.TOOL_RESULT and event.tool_name:
                idx = last_running_idx.pop(event.tool_name, None)
                if idx is not None and tool_calls[idx][1] == "running":
                    tool_calls[idx][1] = "success"

                # Detect file paths in the result string
                for line in (event.text or "").splitlines():
                    candidate = line.strip()
                    if (
                        Path(candidate).suffix in _FILE_EXTENSIONS
                        and Path(candidate).exists()
                        and candidate not in generated_files
                    ):
                        generated_files.append(candidate)

            elif event.type == EventType.TOKEN:
                response_text += event.text or ""

            elif event.type == EventType.AGENT_OUTPUT:
                # Capture the orchestrator's final synthesis when TOKEN streaming
                # is sparse or absent (depends on LLM provider / streaming mode).
                if event.agent_name == "orchestrator" and not response_text and event.text:
                    response_text = event.text

            elif event.type == EventType.WORKFLOW_COMPLETE:
                if not response_text and event.text:
                    response_text = event.text

            history[-1]["content"] = _build_message(
                tool_calls, response_text, generated_files
            )
            yield history, generated_files or None

    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"❌ **Error:** {exc}"
        yield history, None

    finally:
        with _ip_lock:
            _active_ips.discard(ip)


def _clear() -> tuple[list, None]:
    return [], None


# ---------------------------------------------------------------------------
# Provider → default model mapping
# ---------------------------------------------------------------------------

_PROVIDERS = ["anthropic", "openai", "google", "mistral", "groq"]

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
}


def _default_model_for(provider: str) -> str:
    return _DEFAULT_MODELS.get(provider, "")


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_INTRO = """
**Hermes** is a multi-agent AI framework for equity research. Ask it to:

- 📊 Pull and analyze SEC filings — 10-K, 10-Q, insider transactions, institutional holdings
- 📉 Fetch FRED macroeconomic data — GDP, CPI, rates, employment
- 💹 Get live market quotes and historical price data
- 📰 Search financial news and summarise recent catalysts
- 📗 Build Excel financial models — DCF, three-statement, comps, LBO
- 📝 Generate equity research reports as Word/PDF documents
"""

_EXAMPLES = [
    "Give me a quick overview of Apple's latest 10-K",
    "What are the current revenue and EBITDA margins for Microsoft?",
    "Build a simple DCF model for Tesla using their latest 10-K financials",
    "How has the 10-year Treasury yield moved over the last two years?",
    "What are analysts and the press saying about Nvidia recently?",
    "Analyse insider transactions for Amazon in the last 6 months",
]

_env_provider = os.environ.get("HERMES_LLM_PROVIDER", "anthropic")
_env_model = os.environ.get(
    "HERMES_LLM_MODEL", _DEFAULT_MODELS.get(_env_provider, "claude-sonnet-4-6")
)

with gr.Blocks(title="Hermes Financial Research") as demo:
    gr.Markdown("# 🏛 Hermes Financial Research")
    gr.Markdown(_INTRO)

    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            provider_dd = gr.Dropdown(
                choices=_PROVIDERS,
                value=_env_provider,
                label="LLM Provider",
                scale=1,
            )
            model_tb = gr.Textbox(
                value=_env_model,
                label="Model",
                scale=2,
            )
        with gr.Row():
            sec_ua_tb = gr.Textbox(
                value=os.environ.get("HERMES_SEC_USER_AGENT", ""),
                label="SEC User Agent (required)",
                placeholder="YourApp/1.0 you@email.com",
                scale=3,
            )
            fred_key_tb = gr.Textbox(
                value=os.environ.get("HERMES_FRED_API_KEY", ""),
                label="FRED API Key (optional — enables macro tools)",
                placeholder="abcdef1234...",
                scale=2,
                type="password",
            )
        provider_dd.change(fn=_default_model_for, inputs=provider_dd, outputs=model_tb)

    chatbot = gr.Chatbot(
        value=[],
        label="Research Chat",
        height=560,
        render_markdown=True,
        layout="bubble",
        buttons=["copy"],
    )

    with gr.Row():
        msg_tb = gr.Textbox(
            placeholder="Ask about stocks, filings, macro data, or request a model / report…",
            show_label=False,
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    files_out = gr.Files(
        label="📥 Generated Files",
        interactive=False,
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=msg_tb,
        label="Example queries",
    )

    # Helper: gather settings inputs
    _settings = [provider_dd, model_tb, sec_ua_tb, fred_key_tb]

    # Submit via Enter key
    submit = msg_tb.submit(
        fn=respond,
        inputs=[msg_tb, chatbot, *_settings],
        outputs=[chatbot, files_out],
        show_progress="hidden",
    )
    submit.then(fn=lambda: "", outputs=msg_tb)

    # Submit via Send button
    click = send_btn.click(
        fn=respond,
        inputs=[msg_tb, chatbot, *_settings],
        outputs=[chatbot, files_out],
        show_progress="hidden",
    )
    click.then(fn=lambda: "", outputs=msg_tb)

    # Clear button
    clear_btn.click(fn=_clear, outputs=[chatbot, files_out])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(),
    )
