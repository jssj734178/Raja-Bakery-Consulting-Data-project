#!/bin/bash
# Double-click to shut the bakery invoice system down. Nothing is lost:
# everything stays saved and comes back next time you start it.

cd "$(dirname "$0")" || exit 1
# Icons start programs with a very short list of places to look for commands.
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
./"Backup Bakery.command" --quiet || true
docker compose stop
echo
echo "Stopped. You can close this window."
