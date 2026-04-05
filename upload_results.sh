#!/usr/bin/env bash
set -euo pipefail

# Upload MuseTalk results directory to object storage.
# Supports:
# 1) Tar mode (default): create one .tar archive and upload it
# 2) Sync mode: upload all files directly (slower for many small files)
#
# Examples:
#   ./upload_results.sh jaat:MuseTalk/archives --source results/v15/avatars/my_avatar_720_live
#   ./upload_results.sh s3://my-bucket/musetalk/archives --source results/v15/avatars/my_avatar_720_live
#   ./upload_results.sh jaat:MuseTalk/results --mode sync --source results/v15/avatars/my_avatar_720_live
#
# Optional env for S3-compatible endpoints:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL

SOURCE_DIR="results"
DEST=""
MODE="tar"
ARCHIVE_NAME=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <destination> [--source <dir>] [--mode tar|sync] [--archive-name <name.tar>]

Arguments:
  destination         Target directory:
                      - s3://bucket/path
                      - remote:path
  --source <dir>      Source directory to upload (default: results)
  --mode <mode>       Upload mode: tar (default) or sync
  --archive-name      Archive file name in tar mode (default: <source_basename>.tar)
  -h, --help          Show this help

Examples:
  $(basename "$0") jaat:MuseTalk/archives --source results/v15/avatars/my_avatar_720_live
  $(basename "$0") s3://my-bucket/musetalk/archives --source results/v15/avatars/my_avatar_720_live
  $(basename "$0") jaat:MuseTalk/results --mode sync --source results/v15/avatars/my_avatar_720_live
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --archive-name)
      ARCHIVE_NAME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$DEST" ]]; then
        DEST="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$DEST" ]]; then
  echo "Missing destination." >&2
  usage
  exit 1
fi

if [[ "$MODE" != "tar" && "$MODE" != "sync" ]]; then
  echo "Invalid mode: $MODE (use tar or sync)" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Destination: $DEST"
echo "Mode: $MODE"

upload_with_aws() {
  local src="$1"
  local dst="$2"
  if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
    aws s3 cp "$src" "$dst" --endpoint-url "$AWS_ENDPOINT_URL" --no-progress
  else
    aws s3 cp "$src" "$dst" --no-progress
  fi
}

if [[ "$MODE" == "sync" ]]; then
  if [[ "$DEST" == s3://* ]]; then
    if ! command -v aws >/dev/null 2>&1; then
      echo "aws cli not found. Install it first: pip install awscli" >&2
      exit 1
    fi

    if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
      aws s3 sync "$SOURCE_DIR" "$DEST" --endpoint-url "$AWS_ENDPOINT_URL" --no-progress
    else
      aws s3 sync "$SOURCE_DIR" "$DEST" --no-progress
    fi
  else
    if ! command -v rclone >/dev/null 2>&1; then
      echo "rclone not found. Install and configure a remote first: https://rclone.org/" >&2
      exit 1
    fi
    rclone sync "$SOURCE_DIR" "$DEST" --progress
  fi
else
  src_abs="$(cd "$(dirname "$SOURCE_DIR")" && pwd)/$(basename "$SOURCE_DIR")"
  src_base="$(basename "$SOURCE_DIR")"
  if [[ -z "$ARCHIVE_NAME" ]]; then
    ARCHIVE_NAME="${src_base}.tar"
  fi
  case "$ARCHIVE_NAME" in
    *.tar) ;;
    *) ARCHIVE_NAME="${ARCHIVE_NAME}.tar" ;;
  esac

  tmp_archive="$(mktemp "/tmp/${src_base}.XXXXXX.tar")"
  tar -cf "$tmp_archive" -C "$(dirname "$src_abs")" "$src_base"
  target="${DEST%/}/${ARCHIVE_NAME}"
  echo "Archive: $tmp_archive"
  echo "Target: $target"

  if [[ "$DEST" == s3://* ]]; then
    if ! command -v aws >/dev/null 2>&1; then
      echo "aws cli not found. Install it first: pip install awscli" >&2
      rm -f "$tmp_archive"
      exit 1
    fi
    upload_with_aws "$tmp_archive" "$target"
  else
    if ! command -v rclone >/dev/null 2>&1; then
      echo "rclone not found. Install and configure a remote first: https://rclone.org/" >&2
      rm -f "$tmp_archive"
      exit 1
    fi
    rclone copyto "$tmp_archive" "$target" --progress
  fi

  rm -f "$tmp_archive"
fi

echo "Upload complete."
