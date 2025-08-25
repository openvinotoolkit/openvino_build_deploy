from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import math
import os
from .utils import load_documents, setup_models
from utils.custom_embeddings import OpenVINOSentenceTransformerEmbedding
from dotenv import load_dotenv
from llama_index.core import Settings
from pathlib import Path

load_dotenv()

# initialize embedding model
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")

_embedding = OpenVINOSentenceTransformerEmbedding(model_path=str(embedding_model_path), device="CPU")
Settings.embed_model = _embedding

# # initialize llm
# llm = setup_models()
# Settings.llm = llm
Settings.llm = None  # Set to None if no LLM is used

# load document
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")
index = load_documents(Path(DOCUMENT_PATH))



mcp = FastMCP("Smart Retail Tools", port=3001, host="localhost")

# In-memory cart storage
_cart_items = []

@mcp.tool()
async def product_query(query: str) -> str:
    """
    Use this tool for retrieve information about paint products, recommendations, prices, or technical specifications.

    Args:
        query (str): The search query string.

    Returns:
        str: Answer to the query

    WHEN TO USE:
        - User asks about paint types, brands, or products
        - User needs price information before adding to cart
        - User needs recommendations based on their project
        - User has technical questions about painting
        
    EXAMPLES:
        - "What paint is best for kitchen cabinets?"
        - "How much does AwesomePainter Interior Acrylic Latex cost?"
        - "What supplies do I need for painting my living room?"
    """

    retriever = index.as_retriever(verbose=False)
    results = retriever.retrieve(query)
    response = [res.node.get_content() for res in results]
    res = "\n\n".join(response)
    return res

# --- Tool 1: Calculate Paint Cost ---
@mcp.tool()
def calculate_paint_cost(area: float, price_per_gallon: float, add_paint_supply_costs: bool = False) -> float:
    """
    Use this tool to calculate the total cost of paint needed for a given area in square feet.

    Parameters:
    - area (float): Area to paint in square feet.
    - price_per_gallon (float): Price per gallon in USD.
    - add_paint_supply_costs (bool, optional): Whether to add $50 for supplies.

    Returns:
    - Total cost as a float.

    Example:
    calculate_paint_cost(area=600, price_per_gallon=29.99, add_paint_supply_costs=True)
    """
    gallons_needed = math.ceil((area / 400) * 2)
    total_cost = round(gallons_needed * price_per_gallon, 2)
    if add_paint_supply_costs:
        total_cost += 50
    return total_cost

# --- Tool 2: Calculate Gallons Needed ---

@mcp.tool()
def calculate_paint_gallons(area: float) -> str:
    """
    Use this tool to calculate how many gallons of paint are needed to cover a specific area.

    Parameters:
    - area (float): Area in square feet.

    Returns:
    - Number of gallons as a string of an integer (rounded up).

    Example:
    calculate_paint_gallons(area=600)
    """
    return str(math.ceil((area / 400) * 2))

# --- Tool 3: Add to Cart ---

@mcp.tool()
def add_to_cart(product_name: str, quantity: int, price_per_unit: float) -> dict:
    """
    Use this tool to add a product to the shopping cart.

    Parameters:
    - product_name (str): Name of the product.
    - quantity (int): Number of units to add.
    - price_per_unit (float): Unit price in USD.

    Returns:
    - Confirmation message and current cart content.

    Example:
    add_to_cart(product_name="Paintbrush", quantity=2, price_per_unit=5.99)
    """
    item = {
        "product_name": product_name,
        "quantity": quantity,
        "price_per_unit": price_per_unit,
        "total_price": round(quantity * price_per_unit, 2)
    }

    for existing_item in _cart_items:
        if existing_item["product_name"] == product_name:
            existing_item["quantity"] += quantity
            existing_item["total_price"] = round(existing_item["quantity"] * existing_item["price_per_unit"], 2)
            return {
                "message": f"Updated {product_name} quantity to {existing_item['quantity']}",
                "cart": _cart_items
            }

    _cart_items.append(item)
    return {
        "message": f"Added {quantity} {product_name} to cart",
        "cart": _cart_items
    }

# --- Tool 4: View Cart ---
@mcp.tool()
def view_cart() -> list:
    """
    Use this tool to view the current contents of the shopping cart.

    Returns:
    - List of all items in the cart with their details.

    Example:
    view_cart()
    """
    return _cart_items

# --- Tool 5: Clear Cart ---
@mcp.tool()
def clear_cart() -> dict:
    """
    Clear all items from the shopping cart.

    Returns:
    - Confirmation message.

    Example:
    clear_cart()
    """
    _cart_items.clear()
    return {"message": "Shopping cart has been cleared"}
