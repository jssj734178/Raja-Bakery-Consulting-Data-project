#!/bin/bash
# Double-click to shut the bakery invoice system down. Nothing is lost:
# everything stays saved and comes back next time you start it.

cd "$(dirname "$0")" || exit 1
./"Backup Bakery.command" --quiet || true
docker compose stop
echo
echo "Stopped. You can close this window."
