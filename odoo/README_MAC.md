# Bakery invoice system on a Mac

This runs the invoice reader inside Odoo on one Mac. You scan invoices to a
PDF, upload the PDF, check what the software read, and it makes draft
invoices in Odoo.

## Setting it up (once, by whoever installs it - about 20 minutes)

This is the only time anyone uses the Terminal. Afterwards the person who
works with the invoices only double-clicks icons. You need the Mac's admin
password and an internet connection.

1. **Install Docker Desktop.** Go to https://www.docker.com/products/docker-desktop/ ,
   download the version for this Mac (Apple chip or Intel chip - Apple menu >
   About This Mac says which), open the file, drag Docker into Applications,
   open it, and click through its questions (enter the admin password when
   asked). Then in Docker Desktop: Settings (gear) > General > tick **Start
   Docker Desktop when you sign in**. That way the system is already waiting
   when the Mac starts.
2. **Get this folder onto the Mac and run the installer.** Open Terminal
   (Spotlight: type "Terminal"), paste this one line, and press Return:

   ```
   git clone https://github.com/jssj734178/Raja-Bakery-Consulting-Data-project.git bakery && bash bakery/odoo/install_mac.sh
   ```

   (If the Mac asks to install developer tools the first time, say yes, wait,
   and paste the line again. If you downloaded the project as a ZIP instead,
   unzip it and run `bash` followed by the path of `odoo/install_mac.sh`.)
   It builds everything (the long part - mostly downloading) and ends by
   printing **DONE**.
3. **Two icons are now on the Desktop:** *Bakery Invoices* and *Stop Bakery*.
   Drag *Bakery Invoices* into the Dock if you like.
4. **Log in once.** Double-click *Bakery Invoices*. The browser opens the
   system. Log in with the address, login and password in
   `bakery/odoo/Bakery login.txt` and let the browser **remember the
   password**. Change the password in Odoo if you like (top-right menu > My
   Profile).
5. **Check it from a phone** on the bakery Wi-Fi: the address (like
   `http://192.168.1.20:8069`) is shown by the Mac in System Settings > Wi-Fi
   > Details, or run `ipconfig getifaddr en0` in Terminal.

## Every day (no Terminal)

- **Start:** double-click **Bakery Invoices** on the Desktop. It starts Docker
  if it isn't running, starts the system, saves one backup a day, and opens
  it in the browser. If the Mac was just switched on, allow a minute or two.
  Once the system is open, the program that reads the handwriting is just a
  menu inside it - there is nothing separate to open.
- **Use it:** Bakery Invoices > Upload scanned invoices, pick the PDF.
  Each page becomes one invoice, read in the background (about 3 seconds a
  page after the first, which takes about 20 seconds). Open "Scanned
  invoices", open one, check the rows (red rows were flagged, with the
  picture of the handwriting and the reason), fix the customer, date,
  paper invoice number and any wrong numbers, then press **Create and post
  invoice**. (If the paper number is blank, a suggested next number is shown
  with a "Use it" button - it is only a guess, so check it against the
  paper.) "Create draft only" is there if you want to look at it in Odoo
  before posting.
- **Who still owes money:** Bakery Invoices > Unpaid invoices, grouped by
  customer. Odoo only counts an invoice as unpaid once it is posted, which
  is why the main button posts it.
- **When a customer pays (usually months later):** open Unpaid invoices,
  tick the invoices that payment covers, then Actions > Pay, and enter the
  amount and date. They move to Paid and drop off the list.
- **Stop:** double-click **Stop Bakery** when you are done for the day. (Or
  just shut the Mac down: everything is saved and starts again next time.)
- **From a phone or another computer on the bakery Wi-Fi:** use the address
  from setup step 5.

## Backups

`Backup Bakery.command` (in the `odoo` folder) saves everything into a `BakeryBackups` folder in
the home folder, and keeps the newest 14. The start and stop scripts do
this automatically once a day. Copy that folder to a USB drive or cloud
drive now and then - a backup that lives only on the same Mac does not
protect against that Mac being lost.

### Restoring a backup

Only needed to move to a new Mac, or after something goes badly wrong. Set
the system up on the new Mac as above, then in Terminal, from the `odoo`
folder (replace the two file names with the ones you want):

```
docker compose stop odoo
docker compose exec -T db dropdb -U odoo bakery
docker compose exec -T db createdb -U odoo bakery
gunzip -c ~/BakeryBackups/bakery-DATE.sql.gz | docker compose exec -T db psql -U odoo bakery
docker compose run --rm -T odoo bash -c "cd /var/lib/odoo && tar xzf -" < ~/BakeryBackups/bakery-DATE-files.tar.gz
docker compose up -d
```

## Updating to a newer version

Whoever maintains it double-clicks `Update Bakery.command` in the `odoo` folder (needs internet). Your invoices and
settings are not touched.

## If something goes wrong

- **"Docker is not running" / nothing opens:** open Docker Desktop, wait
  until it says it is running, double-click Start again.
- **A page says "Needs manual handling":** the scan's printed table could
  not be found (a very crooked or cropped scan). Rescan it, or make that
  invoice by hand in Odoo.
- **Reading seems stuck:** pages are read one at a time in the background;
  a full PDF of 14 pages takes about a minute after the first start.
  Refresh the "Scanned invoices" list.
- **Forgot the password:** it is in `odoo/Bakery login.txt` unless you
  changed it.
