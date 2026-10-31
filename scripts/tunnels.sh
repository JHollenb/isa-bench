#!/usr/bin/env bash
# macOS blocks Python from the LAN; forward Beast's mrun (9025), MLflow (5000) and cold MinIO (9010) to localhost.
set -euo pipefail
for spec in 19025:127.0.0.1:9025 15000:127.0.0.1:5000 19010:127.0.0.1:9010; do
  port=${spec%%:*}
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then echo "tunnel $port up"; continue; fi
  ssh -fN -o ExitOnForwardFailure=yes -L "$spec" beast && echo "tunnel $port opened"
done
