#!/bin/bash
# Starts the bakery invoice system on a Mac. Normally you never run this by
# hand: the "Bakery Invoices" icon on the Desktop runs it (with --quiet).
# The first run also sets everything up (about 10-15 minutes, mostly
# downloading); after that it starts in under a minute.

cd "$(dirname "$0")" || exit 1
QUIET=0
[ "$1" = "--quiet" ] && QUIET=1

# Programs started from an icon get a very short list of places to look for
# commands, so name where Docker Desktop puts its tools.
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

fail() {
  echo
  echo "PROBLEM: $*"
  if [ "$QUIET" = 0 ]; then
    echo
    read -n 1 -s -r -p "Press any key to close this window."
  fi
  exit 1
}

step() { echo; echo "==> $*"; }

command -v docker >/dev/null 2>&1 || fail "Docker Desktop is not installed yet. Download it from https://www.docker.com/products/docker-desktop/ , install it, open it once, then try again."

if ! docker info >/dev/null 2>&1; then
  step "Starting Docker Desktop (this can take a minute or two)"
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

# Only build when the program has never been built. Day to day this skips the
# build entirely, so starting works even with no internet; "Update Bakery"
# is what rebuilds on purpose.
if ! docker image inspect bakery-odoo:latest >/dev/null 2>&1; then
  step "Building the program (the first time this downloads a lot - please wait)"
  docker compose build || fail "The build failed. Check the internet connection and try again."
fi

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

  # Canadian dollars, and the one login to use. The password is saved in
  # "Bakery login.txt" next to this file.
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
READY=0
for _ in $(seq 1 90); do
  if curl -fs -o /dev/null http://localhost:8069/web/login; then READY=1; break; fi
  sleep 2
done
[ "$READY" = 1 ] || fail "The invoice system started but is not answering yet. Wait a minute and try again."

# One automatic backup per day, whenever the system is started.
./"Backup Bakery.command" --quiet || true

IP=$(ipconfig getifaddr en0 2>/dev/null)
echo
echo "Ready."
echo "From a phone or another computer on the bakery Wi-Fi: http://${IP:-<the Mac address>}:8069"
[ -f "Bakery login.txt" ] && { echo; cat "Bakery login.txt"; }
if [ "$QUIET" = 0 ]; then
  open http://localhost:8069
  echo
  echo "You can close this window. Use 'Stop Bakery' when you are finished for the day."
fi
