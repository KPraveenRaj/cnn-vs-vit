#!/usr/bin/env bash
# Obtain Food-101 from the official ETH Zurich release. Idempotent and validated.
#
# WHY NOT THE LOCAL archive_food.zip:
# The staged wrapper (a Kaggle-style re-upload) is damaged. Its payload member is
# 5,019,619,529 bytes, which Debian's UnZip 6.00 mishandles as a >4 GB zip64
# member; extracting it with Python instead succeeds, but the resulting inner
# archive is itself malformed — its central directory records local-header
# offsets inflated by 2^32 for 71% of members, and the shift is not uniform
# (43 of 51 sampled offsets resolve, the rest do not). Data-descriptor bits are
# also set, so csize is absent from the local headers and a sequential rescan
# cannot reliably recover the remainder either. Roughly 16% of images would go
# missing in an unpredictable pattern, which would quietly bias per-class counts.
#
# The official tarball is the same dataset with sound provenance, so this
# downloads that instead. The broken wrapper is left untouched on disk.
set -euo pipefail
cd "$(dirname "$0")/../data/raw/food101"
URL="http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EXPECT_CLASSES=101
EXPECT_IMAGES=101000

if [ ! -d food-101/images ]; then
  if [ ! -f food-101.tar.gz ]; then
    echo "[food101] downloading official tarball (~5.0 GB, resumable) ..."
    curl -L -C - --no-progress-meter -o food-101.tar.gz "$URL"
  fi
  echo "[food101] extracting ..."
  tar -xzf food-101.tar.gz
fi

CLASSES=$(ls food-101/images 2>/dev/null | wc -l)
IMAGES=$(find food-101/images -name '*.jpg' 2>/dev/null | wc -l)
echo "[food101] classes: $CLASSES  images: $IMAGES"
if [ "$CLASSES" -ne "$EXPECT_CLASSES" ] || [ "$IMAGES" -ne "$EXPECT_IMAGES" ]; then
  echo "[food101] FAILED: expected $EXPECT_CLASSES classes / $EXPECT_IMAGES images" >&2
  exit 1
fi
rm -f food-101.tar.gz
echo "[food101] OK  ($(du -sh food-101 | cut -f1))"
