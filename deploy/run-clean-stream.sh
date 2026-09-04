#!/bin/bash
set -euo pipefail

# Import only the refine credential already managed in the service user's
# shell config; do not execute unrelated interactive shell setup.
while IFS= read -r assignment; do
    case "$assignment" in
        "export ARK_APIKEY="*) eval "$assignment" ;;
    esac
done < /home/ubuntu/.bashrc

exec /home/ubuntu/workspace/audiollm-server/.venv/bin/uvicorn \
    backend.main:app --host 127.0.0.1 --port 8083
