import os
import requests
from langchain_core.tools import tool
from config import GOOGLE_MAPS_API_KEY

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    r = 6371 
    distance = 2 * r * atan2(sqrt(d), sqrt(1 - d))
    return distance

@tool
def search_nearby_locations(product_name: str, user_lat: float, user_lng: float, radius: int = 10500) -> str:
    """
    Search for nearby locations that sell a specific Nestlé product using Google Places API.
    
    Args:
        product_name: Name of the Nestlé product to search for
        user_lat: User's latitude coordinate  
        user_lng: User's longitude coordinate
        radius: Search radius in meters (default 10500m = ~10.5km)
    
    Returns:
        Formatted string with nearby locations, addresses, distances, and status
    """
    try:
        API_KEY = GOOGLE_MAPS_API_KEY
        if not API_KEY:
            return "Google Maps API key not configured."
            
        endpoint_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        
        params = {
            'keyword': f"{product_name} nestlé",
            'location': f'{user_lat},{user_lng}',
            'radius': radius,
            'type': 'store',
            'key': API_KEY
        }

        response = requests.get(endpoint_url, params=params)
        
        if response.status_code != 200:
            return f"Error {response.status_code} while searching for {product_name}: {response.text}"
            
        data = response.json()
        results = data.get('results', [])[:4]  # Limit to top 4 results

        if not results:
            return f"No nearby stores found for {product_name}. You might want to try searching online retailers."

        output = f"**Top locations near you for {product_name.upper()}:**\n\n"
        
        for idx, place in enumerate(results, start=1):
            name = place.get('name', 'N/A')
            address = place.get('vicinity', 'No address found')
            lat2 = place['geometry']['location']['lat']
            lng2 = place['geometry']['location']['lng']
            distance = calculate_distance(user_lat, user_lng, lat2, lng2)

            # Check if place is currently open
            open_now = place.get("opening_hours", {}).get("open_now", None)
            if open_now is True:
                status = '🟢 Open'
            elif open_now is False:
                status = '🔴 Closed'
            else:
                status = '⚪ Hours Unknown'
            
            rating = place.get('rating', 'No rating')
            
            output += (
                f"{idx}. **{name}**\n"
                f"    Address: {address}\n"
                f"    Distance: {distance:.1f} km away\n"
                f"    Rating: {rating}\n"
                f"    Status: {status}\n"
                f"    [View on Google Maps](https://www.google.com/maps/search/?api=1&query={lat2},{lng2})\n\n"
            )
            
        return output.strip()
        
    except Exception as e:
        return f"Error searching for {product_name}: {str(e)}"

@tool
def get_amazon_product_links(product_name: str) -> str:
    """
    Generate Amazon Canada search links for Nestlé products.
    
    Args:
        product_names: List of product names to search for
        
    Returns:
        Formatted string with Amazon search links
    """
    try:
        base_url = "https://www.amazon.ca/s?k="
        output = "** Find these products on Amazon Canada:**\n\n"
        link = f"{base_url}{product_name}"
        output += f"• [**{product_name.upper()}**]({link})\n"

        return output.strip()
        
    except Exception as e:
        return f"Error generating Amazon links: {str(e)}"

@tool  
def search_nestle_knowledge_base(query: str) -> str:
    """
    Search the Nestlé knowledge base for product information, recipes, nutrition facts, and general questions.

    Args:
        query: The search query or question about Nestlé products

    Returns:
        Relevant information from the knowledge base
    """
    try:
        from utils import vector_retriever, graph, graph_retriever_simple
        
        if not vector_retriever or not graph:
            return "Knowledge base not available."
            
        vector_results = vector_retriever.invoke(query)
        vector_content = "\n".join([doc.page_content for doc in vector_results])
        graph_content = graph_retriever_simple(query, graph)
        
        combined_content = f"Vector Results:\n{vector_content}\n\nGraph Results:\n{graph_content}"
        print(f"Graph content: {graph_content}\n Vector content: {vector_content}")
        return combined_content if combined_content.strip() else "No relevant information found in the knowledge base."

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"