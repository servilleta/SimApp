
# Secure Content Security Policy Implementation

## Backend Implementation (FastAPI)
Add to your security middleware:

```python
from fastapi import Request, Response

def build_csp_header():
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'strict-dynamic'",
        "style-src 'self' 'unsafe-hashes' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data: https:",
        "connect-src 'self' https://api.stripe.com",
        "frame-src 'none'",
        "frame-ancestors 'none'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests"
    ]
    return "; ".join(csp_directives)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = build_csp_header()
    return response
```

## Frontend Implementation
1. Remove all inline scripts and styles
2. Use nonces for dynamic scripts: <script nonce="{nonce}">
3. Hash static inline styles: sha256-{hash}
4. Move all JavaScript to external files
