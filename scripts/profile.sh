#!/bin/bash

# Profile the data pipeline
cd Novel_Writer

echo "=== Profiling Data Pipeline ==="
python -m novel_writer.profile \
  --clean \
  --format \
  --config config.yaml

echo "\n=== Profiling Complete ==="
echo "Profile files:"
ls -lh *.prof
