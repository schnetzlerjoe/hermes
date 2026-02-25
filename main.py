"""Quick E2E demo: full agent pipeline in one call.

Usage:
    ANTHROPIC_API_KEY=sk-... uv run python main.py
"""

import logging

logging.basicConfig(level=logging.INFO)

from hermes import Hermes, configure

configure(sec_user_agent="HermesFinancial/0.1 (test@example.com)")

h = Hermes(model="gemini-3-flash-preview")
result = h.invoke("Build me a DCF and investment research report for GO")
print(result["response"])
