#!/usr/bin/env bash
# Self-signed certificate for HTTPS on the VM's LAN address: scripts/make_cert.sh [ip]
set -euo pipefail
cd "$(dirname "$0")/.."
ip="${1:-$(hostname -I | awk '{print $1}')}"
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=voicebot" -addext "subjectAltName=IP:$ip,DNS:localhost"
echo "Wrote certs/cert.pem and certs/key.pem for $ip."
echo "Set WEB_HOST=0.0.0.0, WEB_SSL_CERTFILE=certs/cert.pem, WEB_SSL_KEYFILE=certs/key.pem, then open https://$ip:8000"
