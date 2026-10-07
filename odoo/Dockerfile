# Odoo 19 plus everything the invoice-reading engine needs (PyTorch, OpenCV,
# PyMuPDF), with the engine's code and the trained digit model baked in.
# Built from the repository root, not this folder -- see docker-compose.yml.
FROM odoo:19.0

USER root

# CPU-only PyTorch: much smaller than the default download, and the bakery's
# Mac has no NVIDIA graphics card to use anyway. The same wheel index serves
# both Intel and Apple-silicon (Linux ARM) Macs, since Docker runs Linux inside.
RUN pip3 install --no-cache-dir --break-system-packages \
        torch --index-url https://download.pytorch.org/whl/cpu

# Same versions requirements.txt pins -- the reading thresholds were tuned
# against these, and OpenCV in particular can behave slightly differently
# between versions.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.5.2 pillow==12.3.0 pymupdf==1.28.2 opencv-python-headless==5.0.0.93

# The shared engine. Used straight from this folder (no install step), so its
# data files (calibration, product list, trained model) stay next to the code.
COPY alignment.py data.py digit_reader.py extract_invoice.py model.py pdf_to_images.py /opt/bakery_ml/
COPY template_calibration.json product_rows.json customers.json /opt/bakery_ml/
COPY checkpoints/digit_cnn_finetuned.pt /opt/bakery_ml/checkpoints/digit_cnn_finetuned.pt
ENV PYTHONPATH=/opt/bakery_ml

# Deliberately NOT /mnt/extra-addons: the official Odoo image makes that a
# persistent volume, so Docker keeps the first copy of the module forever and
# later versions baked into the image never reach a running system.
COPY odoo/addons /opt/bakery_addons

USER odoo
