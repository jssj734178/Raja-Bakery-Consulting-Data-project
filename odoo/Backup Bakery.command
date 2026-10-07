#!/bin/bash
# Saves a backup of everything (invoices, customers, products, saved pictures)
# into the BakeryBackups folder in your home folder. Keeps the newest 14.
# Run it by hand any time, or let the start/stop scripts do it once a day.
#
# To use a backup on a new Mac: set the system up, then restore the two files
# with the steps in README_MAC.md ("Restoring a backup").

cd "$(dirname "$0")" || exit 1
# Icons start programs with a very short list of places to look for commands.
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
QUIET=0
[ "$1" = "--quiet" ] && QUIET=1

DEST="$HOME/BakeryBackups"
mkdir -p "$DEST"
TODAY=$(date +%Y-%m-%d)

# Already backed up today? Nothing more to do when run automatically.
if [ "$QUIET" = 1 ] && ls "$DEST"/bakery-"$TODAY"-*.sql.gz >/dev/null 2>&1; then
  exit 0
fi

# Skip quietly if the system isn't running (nothing to back up from).
if ! docker compose exec -T db pg_isready -U odoo >/dev/null 2>&1; then
  [ "$QUIET" = 1 ] || echo "The invoice system is not running - start it first, then back up."
  exit 0
fi

STAMP=$(date +%Y-%m-%d-%H%M)
[ "$QUIET" = 1 ] || echo "Backing up..."
docker compose exec -T db pg_dump -U odoo bakery | gzip > "$DEST/bakery-$STAMP.sql.gz" \
  && docker compose exec -T odoo tar czf - -C /var/lib/odoo . > "$DEST/bakery-$STAMP-files.tar.gz"

# Keep only the newest 14 of each kind.
ls -1t "$DEST"/bakery-*.sql.gz 2>/dev/null | tail -n +15 | while read -r old; do rm -f "$old"; done
ls -1t "$DEST"/bakery-*-files.tar.gz 2>/dev/null | tail -n +15 | while read -r old; do rm -f "$old"; done

[ "$QUIET" = 1 ] || { echo "Saved to $DEST"; echo; read -n 1 -s -r -p "Press any key to close this window."; }
