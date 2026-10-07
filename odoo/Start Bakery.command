#!/bin/bash
# Double-click this to start the bakery invoice system on a Mac.
# The first time, it also sets everything up (about 10-15 minutes, mostly
# downloading). After that it starts in under a minute.

cd "$(dirname "$0")" || exit 1

fail() {
  echo
  echo "PROBLEM: $*"
  echo
  read -n 1 -s -r -p "Press any key to close this window."
  exit 1
}

step() { echo; echo "==> $*"; }

command -v docker >/dev/null 2>&1 || fail "Docker Desktop is not installed yet. Download it from https://www.docker.com/products/docker-desktop/ , install it, open it once, then double-click this file again."

if ! docker info >/dev/null 2>&1; then
  step "Starting Docker Desktop (this can take a minute)"
  open -a Docker
  for _ in $(seq 1 90); do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done
  docker info >/dev/null 2>&1 || fail "Docker Desktop did not start. Open it by hand, wait until it says it is running, then try again."
fi

# Fresh random passwords the first time only. They live in .env, which is
# never uploaded anywhere (it is in .gitignore).
FIRST_RUN_PASSWORD=""
if [ ! -f .env ]; then
  FIRST_RUN_PASSWORD=$(openssl rand -base64 12 | tr -d '/+=' | cut -c1-12)
  {
    echo "DB_PASSWORD=$(openssl rand -hex 16)"
  } > .env
fi

step "Building the program (the first time this downloads a lot - please wait)"
docker compose build || fail "The build failed. Check the internet connection and try again."

step "Starting the database"
docker compose up -d db || fail "Could not start the database."
for _ in $(seq 1 30); do
  docker compose exec -T db pg_isready -U odoo >/dev/null 2>&1 && break
  sleep 2
done

DB_EXISTS=$(docker compose exec -T db psql -U odoo -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='bakery'" 2>/dev/null | tr -d '[:space:]')
if [ "$DB_EXISTS" != "1" ]; then
  step "First-time setup: creating the bakery database (a few minutes)"
  docker compose run --rm -T odoo odoo -d bakery -i base,account,bakery_invoice_import \
    --stop-after-init --no-http --log-level=warn \
    || fail "First-time setup failed."

  # Canadian dollars, and the one login to use. The password is printed once
  # and also saved in "Bakery login.txt" next to this file.
  docker compose run --rm -T -e NEW_PASSWORD="$FIRST_RUN_PASSWORD" odoo odoo shell -d bakery --no-http --log-level=warn <<'PY' >/dev/null
import os
cad = env.ref('base.CAD')
cad.active = True
env.company.currency_id = cad
admin = env.ref('base.user_admin')
admin.write({'login': 'admin', 'password': os.environ['NEW_PASSWORD']})
env.cr.commit()
PY
  if [ -n "$FIRST_RUN_PASSWORD" ]; then
    printf 'Address:  http://localhost:8069\nLogin:    admin\nPassword: %s\n' "$FIRST_RUN_PASSWORD" > "Bakery login.txt"
  fi
fi

step "Starting the invoice system"
docker compose up -d || fail "Could not start the invoice system."
for _ in $(seq 1 60); do
  curl -fs -o /dev/null http://localhost:8069/web/login && break
  sleep 2
done

# One automatic backup per day, whenever the system is started.
./"Backup Bakery.command" --quiet || true

IP=$(ipconfig getifaddr en0 2>/dev/null)
open http://localhost:8069
echo
echo "Ready. The invoice system is open in your browser."
echo "To use it from a phone or another computer on the bakery Wi-Fi, go to: http://${IP:-<this Mac's address>}:8069"
[ -f "Bakery login.txt" ] && { echo; cat "Bakery login.txt"; }
echo
echo "You can close this window. Double-click 'Stop Bakery' when you are finished for the day."
