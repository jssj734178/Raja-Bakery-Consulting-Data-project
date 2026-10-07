# Bakery invoice system on a Mac

This runs the invoice reader inside Odoo on one Mac. You scan invoices to a
PDF, upload the PDF, check what the software read, and it makes draft
invoices in Odoo.

## Setting it up (once, about 15 minutes)

You need the Mac's admin password and an internet connection.

1. **Install Docker Desktop.** Go to https://www.docker.com/products/docker-desktop/ ,
   download the version for this Mac (Apple chip or Intel chip - the page
   has both; Apple menu > About This Mac says which), open the downloaded
   file, drag Docker into Applications, open it, and click through its
   questions. Enter the admin password when it asks.
2. **Get this folder onto the Mac.** Either:
   - open Terminal and type `git clone https://github.com/jssj734178/Raja-Bakery-Consulting-Data-project.git`
     (then the folder is wherever Terminal was opened, usually your home folder), or
   - download the project as a ZIP from GitHub and unzip it. If you do this,
     open Terminal once and type `chmod +x ` (with a space after it), drag
     the four `.command` files from the `odoo` folder into the Terminal
     window, and press Return. The ZIP download loses a setting the
     `.command` files need; `git clone` does not.
3. **Double-click `odoo/Start Bakery.command`.** The first time, macOS may
   say it is from an unidentified developer: right-click the file, choose
   Open, then Open again. It then builds everything (the long part - mostly
   downloading), and opens the system in your browser.
4. **The login** is printed in the window and saved in `odoo/Bakery login.txt`
   (address, `admin`, and a password). Keep that file safe, and change the
   password in Odoo if you like (top-right menu > My Profile).

## Every day

- **Start:** double-click `Start Bakery.command`. It starts Docker if needed
  and opens the system. It also saves one backup per day.
- **Use it:** Bakery Invoices > Upload scanned invoices, pick the PDF.
  Each page becomes one invoice, read in the background (about 3 seconds a
  page after the first, which takes about 20 seconds). Open "Scanned
  invoices", open one, check the rows (red rows were flagged, with the
  picture of the handwriting and the reason), fix the customer, date,
  paper invoice number and any wrong numbers, then press **Create draft
  invoice**. The draft is in Accounting > Customers > Invoices, where a
  person gives it a final look and posts it.
- **Stop:** double-click `Stop Bakery.command` when you are done for the day.
- **From a phone or another computer on the bakery Wi-Fi:** the start
  window prints an address like `http://192.168.1.20:8069`.

## Backups

`Backup Bakery.command` saves everything into a `BakeryBackups` folder in
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

Double-click `Update Bakery.command` (needs internet). Your invoices and
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
