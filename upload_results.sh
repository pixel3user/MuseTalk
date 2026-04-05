#!/usr/bin/env bash
set -euo pipefail

# Upload MuseTalk results directory to object storage.
# Supports:
# 1) S3-compatible backends via aws cli (s3://...)
# 2) Any rclone remote (remote:path)
#
# Examples:
#   ./upload_results.sh s3://my-bucket/musetalk/results
#   ./upload_results.sh s3://my-bucket/musetalk/results --source results/v15/avatars/my_avatar
#   ./upload_results.sh gdrive:musetalk/results
#
# Optional env for S3-compatible endpoints:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL

SOURCE_DIR="results"
DEST=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <destination> [--source <dir>]

Arguments:
  destination         Target path:
                      - s3://bucket/path            (uses aws s3 sync)
                      - remote:path                 (uses rclone sync)
  --source <dir>      Source directory to upload (default: results)
  -h, --help          Show this help

Examples:
  $(basename "$0") s3://my-bucket/musetalk/results
  $(basename "$0") s3://my-bucket/musetalk/results --source results/v15/avatars/my_avatar_720_live
  $(basename "$0") b2:musetalk/results
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

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Destination: $DEST"

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

echo "Upload complete."
