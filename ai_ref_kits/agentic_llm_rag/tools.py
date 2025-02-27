import math

class PaintCalculator:

    @staticmethod
    def calculate_paint_cost(area: float, price_per_gallon: float, add_paint_supply_costs: bool = False) -> float:
        """
        Calculate the total cost of paint needed for a given area.
        
        Args:
            area: Area to be painted in square feet
            price_per_gallon: Price per gallon of paint
            add_paint_supply_costs: Whether to add $50 for painting supplies
            
        Returns:
            Total cost of paint and supplies if requested
        """
        gallons_needed = math.ceil((area / 400) * 2) # Assuming 2 gallons are needed for 400 square feet
        total_cost = round(gallons_needed * price_per_gallon, 2)
        if add_paint_supply_costs:
            total_cost += 50
        return total_cost

    @staticmethod
    def calculate_paint_gallons_needed(area: float) -> int:
        """
        Calculate the number of gallons of paint needed for a given area.
        
        Args:
            area: Area to be painted in square feet
            
        Returns:
            Number of gallons needed (rounded up to ensure coverage)
        """
        # Using the same formula as in PaintCostCalculator: 2 gallons needed for 400 square feet
        gallons_needed = math.ceil((area / 400) * 2)
        return gallons_needed

class ShoppingCart:
    # In-memory shopping cart
    _cart_items = []
    
    @staticmethod
    def add_to_cart(product_name: str, quantity: int, price_per_unit: float) -> dict:
        """
        Add an item to the shopping cart.
        Add a product to a user's shopping cart.
        This function ensures a seamless update to the shopping cart by specifying each required input clearly.
        
        Args:
            product_name: Name of the paint product
            quantity: Number of units/gallons
            price_per_unit: Price per unit/gallon
            
        Returns:
            Dict with confirmation message and current cart items
        """
        item = {
            "product_name": product_name,
            "quantity": quantity,
            "price_per_unit": price_per_unit,
            "total_price": round(quantity * price_per_unit, 2)
        }
        
        # Check if item already exists
        for existing_item in ShoppingCart._cart_items:
            if existing_item["product_name"] == product_name:
                # Update quantity
                existing_item["quantity"] += quantity
                existing_item["total_price"] = round(existing_item["quantity"] * existing_item["price_per_unit"], 2)
                return {
                    "message": f"Updated {product_name} quantity to {existing_item['quantity']} in your cart",
                    "cart": ShoppingCart._cart_items
                }
        
        # Add new item
        ShoppingCart._cart_items.append(item)
        
        return {
            "message": f"Added {quantity} {product_name} to your cart",
            "cart": ShoppingCart._cart_items
        }
    
    @staticmethod
    def get_cart_items() -> list:
        """
        Get all items currently in the shopping cart.
        
        Returns:
            List of items in the cart with their details
        """
        return ShoppingCart._cart_items
    
    @staticmethod
    def clear_cart() -> dict:
        """
        Clear all items from the shopping cart.
        
        Returns:
            Confirmation message
        """
        ShoppingCart._cart_items = []
        return {"message": "Shopping cart has been cleared"}