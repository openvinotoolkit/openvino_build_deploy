class Math(object):

    def add(self, a: float, b: float) -> float:
        """Add two numbers and returns the sum"""
        return a + b
    def subtract(self, a: float, b: float) -> float:
        """Add two numbers and returns the sum"""
        return a - b
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers and returns the product"""
        return a * b
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers and returns the quotient"""
        return a / b


class PaintCostCalculator(object):

    def calculate_paint_cost(self, area: int, price_per_gallon: int, add_paint_supply_costs: bool) -> float:
        """Assuming 2 gallons are needed for 400 square feet"""
        gallons_needed = (area / 400) * 2
        total_cost = gallons_needed * price_per_gallon
        if add_paint_supply_costs is True:
            total_cost += 50
        return total_cost
