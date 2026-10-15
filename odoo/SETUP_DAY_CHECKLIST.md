# Setup-day checklist (for whoever installs it, with or without Claude Code)

If you are Claude Code: first read the section "If you are Claude Code running on
the client's Mac" at the top of CLAUDE.md (what you may and may not change, and
what never to delete). Write your numbers in `MAC_MEASUREMENTS.md` and commit on
the `mac-setup` branch.

Nothing in the Mac scripts has been run on a Mac yet - they only passed a
syntax check. Treat setup day as the first real test and note what fails.

## Already checked before setup day (from the Windows PC)
- The base images `odoo:19.0` and `postgres:15` both publish Apple-silicon (arm64) builds.
- PyTorch 2.14.1 (now pinned in `Dockerfile`), numpy 2.5.2, pillow 12.3.0, pymupdf 1.28.2
  and opencv-python-headless 5.0.0.93 all publish Linux ARM wheels for Python 3.12.
- `Backup Bakery.command` backs up the database AND Odoo's whole data folder (which holds
  the page PDFs and the digit bank). Read from the script; never run on a Mac.

## Before you start
- [ ] Mac: MacBook Air M4, 16 GB, about 95 GB free. Plugged in. Low Power Mode off.
- [ ] Admin password to hand, and internet.

## Install (see README_MAC.md)
- [ ] Docker Desktop installed; "Start Docker Desktop when you sign in" ticked.
- [ ] Docker Desktop > Settings > Resources > Memory about 6 GB.
- [ ] `git clone ... && bash bakery/odoo/install_mac.sh` finishes with DONE.
      If the build fails, note the exact error - the likeliest cause is the PyTorch download.
- [ ] Two Desktop icons exist (Bakery Invoices, Stop Bakery) and Bakery Invoices opens the browser.
- [ ] Logged in once; browser remembers the password.

## Measure (about 15 minutes) - write the numbers down
- [ ] Native build? `docker image inspect bakery-odoo --format '{{.Architecture}}'` must say `arm64`.
      If it says `amd64`, Docker is emulating Intel and pages will be very slow - fix before anything else.
- [ ] Upload a real 3-page PDF. Time page 1 (includes loading the model) and pages 2 and 3.
      Under about 30 s each for pages 2 and 3 is fine; over a minute is worth investigating.
- [ ] While a page reads: `docker stats` - note the peak memory of the Odoo container.
- [ ] Open a page: customer / date / invoice-number pictures show; rows show their pictures.
- [ ] Create a draft invoice, then a posted one; check Unpaid invoices lists the posted one.
- [ ] Run `Backup Bakery.command`; check the files in `~/BakeryBackups` are not empty
      (a `.sql.gz` and a `-files.tar.gz`).
- [ ] Press Stop Bakery, then start again from the icon; the earlier invoices are still there.

## If speed is a problem (only then)
Do these on the Mac (they don't touch the reading code):
- Docker Desktop memory/CPU limits.
Do NOT change reading thresholds or the render size on the Mac: they were tuned against the
96 old scans, which are not on the Mac. Make such changes on the Windows PC, check with
`compare_extractions.py`, then bring them over with `Update Bakery.command`.
Possible code changes to try on the PC first: fewer worker processes (4-6 instead of all
cores), loading the model when Odoo starts.

## Afterwards
- [ ] Tell her she can delete the original PDFs only after confirming the backup holds the
      page PDFs (the files backup is not empty and a restore has been tried at least once).
- [ ] Record the measured numbers (page time, peak memory, odoo-data size) in CLAUDE.md,
      replacing the estimates.
