Order Service API
Base URL: https://api.example.com
GET /api/orders - List orders with optional filters.
POST /api/orders - Create a new order.
GET /api/orders/{order_id} - Read one order by ID.
PATCH /api/orders/{order_id} - Update order status.
DELETE /api/orders/{order_id} - Cancel an order.
Request fields:
- customer_id: string, required
- sku: string, required
- quantity: integer, required
- status: string, optional
Response:
- status: success
- data: order object