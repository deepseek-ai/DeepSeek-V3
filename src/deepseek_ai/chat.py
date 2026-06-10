import requests
from urllib.parse import urlparse

def validate_url(url):
    """
    Validate URL to prevent SSRF attacks.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_metadata(url):
    """
    Get metadata from a given URL.
    """
    if not validate_url(url):
        return "Invalid URL"
    
    # Add additional validation and sanitization as needed
    try:
        response = requests.get(url)
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)