from beeai_framework.adapters.mcp.serve.server import MCPServer, MCPServerConfig, MCPSettings
from beeai_framework.tools import tool
from beeai_framework.tools.types import JSONToolOutput
import os, requests, yaml
from pathlib import Path

# Load configuration from YAML file
def load_config():
    config_path = Path(__file__).parent.parent / "config" / "mcp_tools.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['tourism_search_server']

# Load configuration
TOURISM_CONFIG = load_config()

# Get API credentials from YAML or environment variables
AMAD_CLIENT_ID = os.getenv("AMAD_CLIENT_ID") or TOURISM_CONFIG['credentials']['client_id']
AMAD_CLIENT_SECRET = os.getenv("AMAD_CLIENT_SECRET") or TOURISM_CONFIG['credentials']['client_secret']

def get_token():
    base_url = TOURISM_CONFIG['api']['base_url']
    token_endpoint = TOURISM_CONFIG['api']['token_endpoint']
    
    r = requests.post(
        f"{base_url}{token_endpoint}",
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "client_credentials",
            "client_id": AMAD_CLIENT_ID,
            "client_secret": AMAD_CLIENT_SECRET
        }
    )
    r.raise_for_status()
    return r.json()["access_token"]

@tool
def search_flights(origin: str, destination: str, departure_date: str, adults: int = 1) -> JSONToolOutput:
    """Search for flight offers using Amadeus API.
    
    Args:
        origin: Origin airport code (e.g., NYC, PAR)
        destination: Destination airport code (e.g., NYC, PAR)
        departure_date: Departure date in YYYY-MM-DD format
        adults: Number of adult passengers (default: 1)
    
    Returns:
        JSONToolOutput: Flight search results from Amadeus API
    """
    try:
        token = get_token()
        base_url = TOURISM_CONFIG['api']['base_url']
        flights_endpoint = TOURISM_CONFIG['api']['flights_endpoint']
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": 1,
            "nonStop": "false",
            "max": 3
        }
        r = requests.get(
            f"{base_url}{flights_endpoint}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.amadeus+json"},
            params=params
        )
        r.raise_for_status()
        return JSONToolOutput(result=r.json())
    except Exception as e:
        return JSONToolOutput(result={"error": f"Error searching flights: {str(e)}"})

@tool
def list_hotels(city_code: str) -> JSONToolOutput:
    """Get top 10 hotel IDs for a city"""
    try:
        token = get_token()
        endpoint = TOURISM_CONFIG['api']['hotels_list_endpoint']
        r = requests.get(
            f"{TOURISM_CONFIG['api']['base_url']}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params={"cityCode": city_code}
        )
        r.raise_for_status()
        data = r.json()
        # Slice to top 10 hotels
        if "data" in data:
            data["data"] = data["data"][:3]
        return JSONToolOutput(result=data)
    except Exception as e:
        return JSONToolOutput(result={"error": f"Error listing hotels: {str(e)}"})

@tool
def search_hotels(hotel_id: str, checkin: str, checkout: str, adults: int = 1) -> JSONToolOutput:
    """Get hotel offers/prices for a specific hotel"""
    try:
        token = get_token()
        endpoint = TOURISM_CONFIG['api']['hotels_offer_endpoint']  # /v3/shopping/hotel-offers
        params = {
            "hotelIds": hotel_id,
            "checkInDate": checkin,
            "checkOutDate": checkout,
            "adults": adults
        }
        r = requests.get(
            f"{TOURISM_CONFIG['api']['base_url']}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params=params
        )
        
        if r.status_code == 200:
            return JSONToolOutput(result=r.json())
        elif r.status_code == 400:
            # Handle "no rooms available" gracefully
            error_data = r.json()
            if any(error.get("code") == 3664 for error in error_data.get("errors", [])):
                return JSONToolOutput(result={
                    "status": "no_availability",
                    "hotel_id": hotel_id,
                    "message": f"No rooms available for {checkin} to {checkout}",
                    "dates": {"checkin": checkin, "checkout": checkout}
                })
            else:
                return JSONToolOutput(result={"error": f"Hotel search error: {error_data}"})
        else:
            r.raise_for_status()
    except Exception as e:
        return JSONToolOutput(result={"error": f"Error searching hotel offers: {str(e)}"})


def main():
    # Load server configuration from YAML
    server_name = TOURISM_CONFIG['name']
    server_port = TOURISM_CONFIG['port']
    server_transport = TOURISM_CONFIG['transport']
    
    # Create MCP server with configuration from YAML
    server = MCPServer(config=MCPServerConfig(
        transport=server_transport,
        settings=MCPSettings(port=server_port),
        name=server_name
    ))
    server.register_many([list_hotels,search_hotels, search_flights])
    print(f"{server_name} starting on port {server_port}\nAvailable tools: {', '.join([t.name for t in [list_hotels,search_hotels, search_flights]])}")
    server.serve()

if __name__ == "__main__":
    main()
