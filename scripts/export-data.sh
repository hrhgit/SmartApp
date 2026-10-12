#!/bin/zsh
# Export an immutable context snapshot into the standard personal data directory.
set -euo pipefail
umask 077

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    print 'Usage: ./scripts/export-data.sh'
    print 'Exports to ~/Library/Application Support/SmartApp/exports/context-v2/<UTC>-<ID>/'
    print 'For a custom destination, use build/smartapp-cli export --output NEW_DIRECTORY.'
    exit 0
  fi
  print -u2 'Unexpected arguments. Run ./scripts/export-data.sh --help.'
  exit 2
fi

project_dir="${0:A:h:h}"
collector_cli="$project_dir/build/smartapp-cli"
if [[ ! -x "$collector_cli" ]]; then
  print -u2 'Build the CLI first: ./scripts/build-app.sh release'
  exit 1
fi

export_root="$HOME/Library/Application Support/SmartApp/exports/context-v2"
mkdir -p "$export_root"
chmod 700 "$export_root" "${export_root:h}"
snapshot_time="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot_id="$(/usr/bin/uuidgen)"
export_dir="$export_root/$snapshot_time-${snapshot_id:l}"

"$collector_cli" export --output "$export_dir"
(
  cd "$export_dir"
  /usr/bin/shasum -a 256 events.jsonl target-history.jsonl navigation-samples.jsonl manifest.json > SHA256SUMS
)
print "Saved snapshot: $export_dir"
