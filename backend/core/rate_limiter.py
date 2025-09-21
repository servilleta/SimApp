from slowapi import Limiter
from slowapi.util import get_remote_address
 
# Initialize the Limiter with default limits directly
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per minute"]) 