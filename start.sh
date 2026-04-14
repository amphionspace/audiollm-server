#!/bin/bash
set -e

cd "$(dirname "$0")"

PORT="${PORT:-8443}"

# Generate self-signed certificate if not present
if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    echo "Generating self-signed SSL certificate..."
    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
        -days 365 -nodes -subj "/CN=audio-demo" 2>/dev/null
    echo "Certificate generated: cert.pem / key.pem"
fi

echo ""
echo "======================================"
echo "  AudioLLM Server"
echo "  https://0.0.0.0:${PORT}"
echo "======================================"
echo ""
echo "  Open in browser:  https://<your-server-ip>:${PORT}"
echo "  First visit: click 'Advanced' -> 'Proceed' to accept self-signed cert"
echo ""

uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem
