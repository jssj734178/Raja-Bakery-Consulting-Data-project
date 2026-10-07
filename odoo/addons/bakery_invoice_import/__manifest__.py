{
    "name": "Bakery Invoice Import",
    "summary": "Upload scanned paper invoices, check what was read, and create draft customer invoices.",
    "version": "19.0.1.0.0",
    "category": "Accounting",
    "author": "Raja Bakery",
    "license": "LGPL-3",
    "depends": ["account", "mail"],
    "data": [
        "security/ir.model.access.csv",
        "wizard/upload_views.xml",
        "views/invoice_scan_views.xml",
        "data/cron.xml",
    ],
    "post_init_hook": "post_init_hook",
    "application": True,
}
