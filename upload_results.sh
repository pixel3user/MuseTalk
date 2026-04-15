#!/usr/bin/env bash
set -euo pipefail

# Upload MuseTalk results directory to object storage and update download script.
# Supports:
# 1) Tar mode (default): create one .tar.gz archive and upload it
# 2) Sync mode: upload all files directly (slower for many small files)
#
# Examples:
#   ./upload_results.sh gs://my-firebase-bucket/results --source results/my_run
#
# Before running for the first time, ensure you are authenticated with the correct account:
#   gcloud auth login
#   gcloud config set account YOUR_EMAIL@gmail.com

SOURCE_DIR="results"
DEST=""
MODE="tar"
ARCHIVE_NAME=""
DOWNLOAD_SCRIPT="download_weights.sh"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <destination> [--source <dir>] [--mode tar|sync] [--archive-name <name.tar.gz>]

Arguments:
  destination         Target directory, MUST be a Firebase/Google Cloud Storage path:
                      - gs://bucket/path
  --source <dir>      Source directory to upload (default: results)
  --mode <mode>       Upload mode: tar (default) or sync
  --archive-name      Archive file name in tar mode (default: <source_basename>_YYYYMMDD_HHMMSS.tar.gz)
  -h, --help          Show this help

Examples:
  $(basename "$0") gs://my-firebase-bucket/results --source results/my_run
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

if [[ "$DEST" != gs://* ]]; then
    echo "Destination must be a gs:// path for Firebase/Google Cloud Storage." >&2
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

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud CLI not found. Install and configure it first: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Destination: $DEST"
echo "Mode: $MODE"

if [[ "$MODE" == "sync" ]]; then
  gcloud storage rsync "$SOURCE_DIR" "$DEST"
  echo "Sync complete. Note: Public URL generation is only supported for 'tar' mode."
else
  # TAR MODE
  src_abs="$(cd "$(dirname "$SOURCE_DIR")" && pwd)/$(basename "$SOURCE_DIR")"
  src_base="$(basename "$SOURCE_DIR")"
  if [[ -z "$ARCHIVE_NAME" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_NAME="${src_base}_${timestamp}.tar.gz"
  fi
  case "$ARCHIVE_NAME" in
    *.tar.gz) ;;
    *) ARCHIVE_NAME="${ARCHIVE_NAME}.tar.gz" ;;
  esac

  tmp_archive="$(mktemp "/tmp/${src_base}.XXXXXX.tar.gz")"
  echo "Creating archive: $tmp_archive"
  tar -czf "$tmp_archive" -C "$(dirname "$src_abs")" "$src_base"

  target_path="${DEST%/}/${ARCHIVE_NAME}"
  echo "Uploading to: $target_path"

  gcloud storage cp "$tmp_archive" "$target_path"

  echo "Setting public read access..."
  gcloud storage objects update "$target_path" --add-acl-grant=entity=AllUsers,role=READER

  bucket_name=$(echo "$DEST" | sed -e 's,gs://,,' -e 's,/.*$,,')
  object_path=$(echo "$target_path" | sed -e "s,gs://${bucket_name}/,,")

  # URL encode the object path
  encoded_object_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$object_path''', safe=''))")

  download_url="https://firebasestorage.googleapis.com/v0/b/${bucket_name}/o/${encoded_object_path}?alt=media"

  echo "========================================================================"
  echo "Upload complete! Download URL:"
  echo "$download_url"
  echo "========================================================================"

  if [[ -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "Updating ${DOWNLOAD_SCRIPT} with new URL..."
    # Use a different delimiter for sed since the URL contains slashes
    sed -i.bak "s|^RESULTS_ARCHIVE_URL=.*|RESULTS_ARCHIVE_URL=\"\${RESULTS_ARCHIVE_URL:-${download_url}}\"|" "$DOWNLOAD_SCRIPT"
    echo "${DOWNLOAD_SCRIPT} has been updated."
    rm -f "${DOWNLOAD_SCRIPT}.bak"
  else
    echo "Warning: ${DOWNLOAD_SCRIPT} not found, could not update URL."
  fi

  rm -f "$tmp_archive"
fi

echo "Script finished."
