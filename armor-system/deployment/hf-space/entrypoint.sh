#!/bin/sh
# Pull the weights, then hand the process over to the service. exec so the
# server is PID 1 and receives the Space's stop signal directly.
set -e
python -I /app/fetch_checkpoints.py
exec python -I /app/armor-system/models/model1_v2/serving/server.py
