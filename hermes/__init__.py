"""hermes-financial: Multi-agent financial research framework.

Quick start::

    from hermes import Hermes, configure

    configure(sec_user_agent="YourApp you@example.com")
    h = Hermes()
    result = h.invoke("Build me a DCF for AAPL")
    print(result["response"])
"""

from hermes.config import HermesConfig, configure
from hermes.core import Hermes
from hermes.infra.streaming import EventType, StreamEvent

__version__ = "0.1.0"

__all__ = [
    "EventType",
    "Hermes",
    "HermesConfig",
    "StreamEvent",
    "__version__",
    "configure",
]
