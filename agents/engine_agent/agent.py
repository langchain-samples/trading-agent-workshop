"""Minimal 'engine-agent' for the LangSmith Engine module (Module 5).

Ships with a deliberate, obvious behavioral bug so LangSmith Engine has a clear,
recurring failure to detect, diagnose, and fix: the calculator tool's `multiply`
branch returns the SUM instead of the product, so every multiplication answer is
wrong. Engine reads the tool source and proposes a one-line code fix.

Keep the bug in place so Engine re-detects it on each fresh project (see the
"Close the loop" section of modules/05_engine.ipynb to fix it for real).
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from utils.models import model


@tool(parse_docstring=True)
def calculator(a: float, b: float, operation: str) -> float:
    """Perform an arithmetic operation on two numbers.

    Args:
        a: The first operand.
        b: The second operand.
        operation: One of "add", "subtract", "multiply", "divide".
    """
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a + b  # BUG: multiplication returns the sum instead of the product.
    if operation == "divide":
        return a / b
    raise ValueError(f"Unknown operation: {operation!r}")


SYSTEM_PROMPT = (
    "You are a calculator assistant. For ANY arithmetic, you must call the "
    "`calculator` tool and report its result exactly. Never compute arithmetic "
    "yourself and never second-guess the tool's output."
)

agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt=SYSTEM_PROMPT,
)
