#!/bin/bash

# Generate SSL certificates for HTTPS
# For production, replace with proper certificates from Let's Encrypt or CA

echo "🔐 Generating SSL certificates for HTTPS..."

# Create SSL directory
mkdir -p ssl/certs ssl/private

# Generate private key
openssl genrsa -out ssl/private/nginx-selfsigned.key 4096

# Generate certificate signing request
openssl req -new -key ssl/private/nginx-selfsigned.key \
    -out ssl/certs/nginx-selfsigned.csr \
    -subj "/C=US/ST=Demo/L=Demo/O=MonteCarloDemo/OU=IT/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 \
    -in ssl/certs/nginx-selfsigned.csr \
    -signkey ssl/private/nginx-selfsigned.key \
    -out ssl/certs/nginx-selfsigned.crt \
    -extensions v3_req \
    -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
)

# Set permissions
chmod 600 ssl/private/nginx-selfsigned.key
chmod 644 ssl/certs/nginx-selfsigned.crt

# Remove CSR file
rm ssl/certs/nginx-selfsigned.csr

echo "✅ SSL certificates generated:"
echo "   Certificate: ssl/certs/nginx-selfsigned.crt"
echo "   Private Key: ssl/private/nginx-selfsigned.key"
echo ""
echo "⚠️  IMPORTANT: These are self-signed certificates for development only!"
echo "   For production, use certificates from Let's Encrypt or a trusted CA."
echo ""
echo "🌐 You can now access the application at:"
echo "   https://localhost (with SSL warning)"
echo "   http://localhost (will redirect to HTTPS)" 