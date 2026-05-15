# Doc2Prototype MVP Run

## Command

`main.py examples/api_doc_sample.png --task api_doc --device CPU --output-dir outputs/mvp_api_image_smoke`

## Input

- Task: `api_doc`
- Source: `examples/api_doc_sample.png`
- Parser: `PaddleOCR-VL OpenVINO`
- OpenVINO device: `CPU`
- OpenVINO version: `2026.1.0-21367-63e31528c62-releases/2026/1`
- OpenVINO model: `ov_paddleocr_vl_model`

## Timings

| Stage | Seconds |
| --- | ---: |
| model_load | 3.862 |
| openvino_inference | 18.830 |
| structure_extraction | 0.001 |
| generation | 0.000 |
| total | 22.988 |

## Structured Output

- Schema version: `doc2prototype.mvp.v1`
- Document type: `api_doc`
- Endpoints: `5`

```json
{
  "schema_version": "doc2prototype.mvp.v1",
  "document_type": "api_doc",
  "title": "Extracted API Documentation",
  "base_url": "",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/orders",
      "description": "List orders with optional filters.",
      "parameters": [],
      "response": {
        "status": "success",
        "data": "object"
      }
    },
    {
      "method": "POST",
      "path": "/api/orders",
      "description": "Create a new order.",
      "parameters": [],
      "response": {
        "status": "success",
        "data": "object"
      }
    },
    {
      "method": "GET",
      "path": "/api/orders/{order_id}",
      "description": "Read one order by ID.",
      "parameters": [
        {
          "name": "order_id",
          "type": "string",
          "required": true,
          "location": "path",
          "description": "Path parameter `order_id`."
        }
      ],
      "response": {
        "status": "success",
        "data": "object"
      }
    },
    {
      "method": "PATCH",
      "path": "/api/orders/{order_id}",
      "description": "Update order status.",
      "parameters": [
        {
          "name": "order_id",
          "type": "string",
          "required": true,
          "location": "path",
          "description": "Path parameter `order_id`."
        }
      ],
      "response": {
        "status": "success",
        "data": "object"
      }
    },
    {
      "method": "DELETE",
      "path": "/api/orders/{order_id}",
      "description": "Cancel an order.",
      "parameters": [
        {
          "name": "order_id",
          "type": "string",
          "required": true,
          "location": "path",
          "description": "Path parameter `order_id`."
        }
      ],
      "response": {
        "status": "success",
        "data": "object"
      }
    }
  ],
  "raw_analysis": "Order Service API\nBase URL: https://api.example.com\nGET /api/orders - List orders with optional filters.\nPOST /api/orders - Create a new order.\nGET /api/orders/{order_id} - Read one order by ID.\nPATCH /api/orders/{order_id} - Update order status.\nDELETE /api/orders/{order_id} - Cancel an order.\nRequest fields:\n- customer_id: string, required\n- sku: string, required\n- quantity: integer, required\n- status: string, optional\nResponse:\n- status: success\n- data: order object"
}
```

## Generated Artifact

- Type: `api_skeleton`
- Generation backend: `deterministic template`

```
"""Auto-generated FastAPI skeleton from Doc2Prototype."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Extracted API Documentation", version="1.0.0")


# ---- Models ----


class GetApiOrdersResponse(BaseModel):
    """Response model for /api/orders."""
    status: str = "success"
    data: Optional[dict] = None



class PostApiOrdersResponse(BaseModel):
    """Response model for /api/orders."""
    status: str = "success"
    data: Optional[dict] = None



class GetApiOrdersOrderIdResponse(BaseModel):
    """Response model for /api/orders/{order_id}."""
    status: str = "success"
    data: Optional[dict] = None



class PatchApiOrdersOrderIdResponse(BaseModel):
    """Response model for /api/orders/{order_id}."""
    status: str = "success"
    data: Optional[dict] = None



class DeleteApiOrdersOrderIdResponse(BaseModel):
    """Response model for /api/orders/{order_id}."""
    status: str = "success"
    data: Optional[dict] = None


# ---- Endpoints ----


@app.get("/api/orders")
async def get_api_orders():
    """
    List orders with optional filters.
    """
    # TODO: Implement business logic
    return {"status": "success", "message": "Not implemented yet"}


@app.post("/api/orders")
async def post_api_orders(payload: Optional[dict] = None):
    """
    Create a new order.
    """
    # TODO: Implement business logic
    return {"status": "success", "message": "Not implemented yet"}


@app.get("/api/orders/{order_id}")
async def get_api_orders_order_id(order_id: str):
    """
    Read one order by ID.
    """
    # TODO: Implement business logic
    return {"status": "success", "message": "Not implemented yet"}


@app.patch("/api/orders/{order_id}")
async def patch_api_orders_order_id(order_id: str, payload: Optional[dict] = None):
    """
    Update order status.
    """
    # TODO: Implement business logic
    return {"status": "success", "message": "Not implemented yet"}


@app.delete("/api/orders/{order_id}")
async def delete_api_orders_order_id(order_id: str):
    """
    Cancel an order.
    """
    # TODO: Implement business logic
    return {"status": "success", "message": "Not implemented yet"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```
