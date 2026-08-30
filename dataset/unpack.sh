#!/bin/sh
# Reassemble the split census archive and extract it into ./data (run from repo root).
set -e
cat dataset/census-data.tar.gz.part* | tar xzf -
echo "Census dataset extracted into data/."
