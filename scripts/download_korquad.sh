#!/usr/bin/env bash
# KorQuAD 1.0 dev 세트를 받는다 (CC BY-ND 2.0 KR — 원문은 커밋하지 않는다).
# 사용법: scripts/download_korquad.sh
set -euo pipefail

BASE_URL="https://korquad.github.io/dataset"
FILE="KorQuAD_v1.0_dev.json"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/korquad"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [[ -f "$FILE" ]]; then
  echo "이미 있음: $DEST_DIR/$FILE — 다운로드 건너뜀"
else
  echo "다운로드: $BASE_URL/$FILE"
  curl -fL --retry 3 -o "$FILE.part" "$BASE_URL/$FILE"
  mv "$FILE.part" "$FILE"
fi
echo "완료: $DEST_DIR/$FILE"
