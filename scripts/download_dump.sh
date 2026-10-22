#!/usr/bin/env bash
# 한국어 위키백과 덤프를 받아 공식 SHA-1 과 대조한다.
# 사용법: scripts/download_dump.sh [덤프 날짜, 기본 20261001]
set -euo pipefail

DUMP_DATE="${1:-20261001}"
FILE="kowiki-${DUMP_DATE}-pages-articles.xml.bz2"
BASE_URL="https://dumps.wikimedia.org/kowiki/${DUMP_DATE}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/dump"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [[ -f "$FILE" ]]; then
  echo "이미 있음: $DEST_DIR/$FILE — 다운로드 건너뜀"
else
  echo "다운로드: $BASE_URL/$FILE"
  curl -fL --retry 3 -C - -o "$FILE.part" "$BASE_URL/$FILE"
  mv "$FILE.part" "$FILE"
fi

echo "체크섬 확인"
expected="$(curl -fsS "$BASE_URL/kowiki-${DUMP_DATE}-sha1sums.txt" | awk -v f="$FILE" '$2 == f {print $1}')"
if [[ -z "$expected" ]]; then
  echo "공식 체크섬 목록에서 $FILE 을 찾지 못함" >&2
  exit 1
fi
actual="$(sha1sum "$FILE" | awk '{print $1}')"
if [[ "$expected" != "$actual" ]]; then
  echo "체크섬 불일치: expected=$expected actual=$actual" >&2
  exit 1
fi
echo "체크섬 일치: $actual"
