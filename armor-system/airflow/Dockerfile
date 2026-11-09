# Airflow image for the ARMOR D1 data pipeline.
#
# The pipeline's dependencies are installed into a SEPARATE virtualenv
# (/opt/pipeline-venv) rather than into Airflow's own environment. The project
# pins pandas 3.x / numpy 2.5.x; Airflow constrains both. Installing them
# together resolves to something neither side tested. Keeping them apart means
# the DAG runs the pipeline against exactly the versions the project pinned.
FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# /opt is root-owned, so the venv directory has to be created and handed over
# before dropping privileges. Building it as root instead would leave root-owned
# files that the airflow user cannot update.
RUN mkdir -p /opt/pipeline-venv && chown -R airflow:0 /opt/pipeline-venv

USER airflow

COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN python -m venv /opt/pipeline-venv \
 && /opt/pipeline-venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/requirements-pipeline.txt \
 && /opt/pipeline-venv/bin/python -c "import numpy, pandas, scipy, matplotlib; print('pipeline venv ok')"
