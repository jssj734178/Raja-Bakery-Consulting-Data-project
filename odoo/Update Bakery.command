#!/bin/bash
# Double-click to install the newest version of the program (needs internet).
# Your invoices and settings are not touched.

cd "$(dirname "$0")" || exit 1
# Icons start programs with a very short list of places to look for commands.
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
./"Backup Bakery.command" --quiet || true
git pull --ff-only || { echo "Could not download the update."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
docker compose build \
  && docker compose up -d \
  && docker compose run --rm -T odoo odoo -d bakery -u bakery_invoice_import --stop-after-init --no-http --log-level=warn
docker compose restart odoo
echo
echo "Updated. You can close this window."
