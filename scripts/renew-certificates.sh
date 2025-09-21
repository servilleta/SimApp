#!/bin/bash
echo "Renewing SSL certificates..."
docker-compose -f docker-compose.domain.yml run --rm certbot renew
docker-compose -f docker-compose.domain.yml restart nginx
echo "Certificate renewal completed"
