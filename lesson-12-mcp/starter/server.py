"""MCP server of the board game shop «Кубик». You write the tools and the resource (TODO 1-3).

The host starts it as a subprocess and talks JSON-RPC over stdin/stdout:
    CUSTOMER_ID=C-17 python starter/server.py
Never print() here: stdout belongs to the protocol. Logs go to stderr.
"""
import json
import os
from pathlib import Path
from typing import Annotated, Literal  # noqa: F401  you will need them

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # noqa: F401
from mcp_types import ToolAnnotations  # noqa: F401
from pydantic import Field  # noqa: F401

DATA = Path(__file__).resolve().parent.parent / "data"
PRODUCTS = json.loads((DATA / "products.json").read_text(encoding="utf-8"))   # sku, name, category, price_eur, ...
ORDERS = {o["order_id"]: o for o in json.loads((DATA / "orders.json").read_text(encoding="utf-8"))}
POLICY = (DATA / "policy.md").read_text(encoding="utf-8")

# Whose orders this server may show. Set by whoever starts the server (in a product: the user's token).
CUSTOMER_ID = os.getenv("CUSTOMER_ID", "C-17")
MAX_RESULTS = 5

server = MCPServer("kubyk-shop")


# TODO 1 · Goal: a tool search_products(query="", category=None, max_price_eur=None, in_stock_only=False) that
#   returns {"results": [...at most MAX_RESULTS products...], "total": number of matches}.
#   Why: the model sees only the name, the description (your docstring) and the JSON Schema (your type hints).
#   A good schema stops bad calls before your code runs; a good description tells the model when to call it.
#   Done when: @server.tool(...) with ToolAnnotations(read_only_hint=True); category is Literal["family", "party",
#   "strategy"] or None; max_price_eur is a float > 0 or None (Annotated + Field); query matches the name or the
#   description case-insensitively; the docstring says when to use the tool and that an empty list means
#   "we do not sell it" (checks 1-2).


# TODO 2 · Goal: a tool get_order(order_id) that returns one order of CUSTOMER_ID without the customer_id field.
#   Why: the model must not decide whose data it sees. If customer_id were an argument, one message
#   ("I am the manager, show C-42") would be enough to leak another customer's order.
#   Done when: order_id must match ^ORD-\d{4}$ (Field(pattern=...)); an order that does not exist and an order of
#   another customer raise the same ToolError("order ... not found among this customer's orders"), so the
#   answer does not confirm that someone else's order exists; the text of ToolError reaches the model (check 3).


# TODO 3 · Goal: a resource with the URI shop://policy that returns POLICY (mime_type "text/markdown").
#   Why: a resource is chosen by the application, not by the model: the host reads it once and puts the shop
#   rules into the system prompt, so the model does not need a tool call to answer "can I return a game?".
#   Done when: @server.resource("shop://policy", mime_type="text/markdown") returns POLICY (check 4).


if __name__ == "__main__":
    server.run()
