#!/bin/bash
# SSL Certificate Initialization Script
# Usage: ./init_ssl.sh your-domain.com

set -e

DOMAIN=${1:-""}
EMAIL=${2:-"admin@example.com"}

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain> [email]"
    echo "Example: $0 api.example.com admin@example.com"
    exit 1
fi

echo "Initializing SSL certificates for $DOMAIN..."

# Create necessary directories
mkdir -p nginx/certbot-data
mkdir -p nginx/certbot-www

# Start certbot in standalone mode to get initial certificate
docker run -it --rm \
    -v "$(pwd)/nginx/certbot-data:/etc/letsencrypt" \
    -v "$(pwd)/nginx/certbot-www:/var/www/certbot" \
    -p 80:80 \
    certbot/certbot:latest \
    certonly \
    --standalone \
    --preferred-challenges http \
    --agree-tos \
    --no-eff-email \
    --email "$EMAIL" \
    -d "$DOMAIN"

echo "SSL certificates initialized successfully!"
echo ""
echo "To start services with SSL:"
echo "  docker compose -f docker-compose.yml -f docker-compose.ssl.yml --profile ssl up -d"
echo ""
echo "To renew certificates manually:"
echo "  docker compose -f docker-compose.ssl.yml run --rm certbot renew"
