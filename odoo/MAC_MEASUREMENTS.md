# Measurements and findings from the Mac (to be filled in on setup day)

Written by Claude Code (or a person) on the client's Mac, committed on the
`mac-setup` branch for the Windows PC session to review. Rules for what may be
changed on the Mac are in the section "If you are Claude Code running on the
client's Mac" at the top of CLAUDE.md. Real numbers only - write "not measured"
rather than guessing. Never put passwords, scans or invoice pictures here.

## Machine
- Date:
- Mac: (expected MacBook Air M4, 16 GB) / macOS version:
- Docker Desktop version, memory setting:
- Plugged in? Low Power Mode off?

## Install
- `install_mac.sh` result (DONE / failed at ...):
- Exact error text if it failed, and what fixed it:
- Desktop icons work (Bakery Invoices, Stop Bakery)? Start from the icon after a restart of the Mac?

## Architecture and speed
- Image architecture (`docker image inspect bakery-odoo --format '{{.Architecture}}'`):
- 3-page upload: page 1 = ___ s (includes model load), page 2 = ___ s, page 3 = ___ s
- Peak memory of the Odoo container during a read (`docker stats`):
- Idle memory (Odoo + database):
- Size of the Odoo data volume right now:

## Backup
- `Backup Bakery.command` run? Size of the `.sql.gz` and `-files.tar.gz`:
- Restore tried (steps in README_MAC.md)? Result:

## Changes tried (one block each)
- What was changed (file, setting):
- Page time before / after:
- Same 3 pages read before and after: all Qty, Return, Total Price identical? (yes / no / not checked)
- Kept or reverted, and why:

## Things that failed or were not tested
-

## Ideas that need the Windows PC (reader or accuracy changes)
-
