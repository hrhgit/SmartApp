#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
app_dir="${1:-$project_dir/build/SmartApp.app}"
bundle_identifier="local.herh.SmartApp"

if [[ ! -d "$app_dir" ]]; then
  print -u2 "App bundle not found: $app_dir"
  exit 1
fi

signing_identity="$($project_dir/scripts/local-signing-identity.sh)"
expected_certificate_hash="${signing_identity:l}"
requirements="$(codesign -d -r- "$app_dir" 2>&1)"

if [[ "$requirements" != *"identifier \"${bundle_identifier}\""* ]]; then
  print -u2 "The app's designated requirement does not contain the fixed bundle identifier."
  exit 1
fi

if [[ "$requirements" != *"certificate root = H\"${expected_certificate_hash}\""* ]]; then
  print -u2 "The app's designated requirement is not anchored to the fixed local signing certificate."
  exit 1
fi

if [[ "$requirements" == *"cdhash"* ]]; then
  print -u2 "The app's designated requirement is tied to a build-specific cdhash."
  exit 1
fi

codesign --verify --strict "$app_dir"
directory_host="$app_dir/Contents/MacOS/smartapp-directory-host"
if [[ -e "$directory_host" ]]; then
  host_requirements="$(codesign -d -r- "$directory_host" 2>&1)"
  if [[ "$host_requirements" != *'identifier "local.herh.SmartApp.DirectoryHost"'* ||
        "$host_requirements" != *"certificate root = H\"${expected_certificate_hash}\""* ]]; then
    print -u2 "The native directory reader is not signed with its fixed identity."
    exit 1
  fi
  codesign --verify --strict "$directory_host"
fi
print "Verified fixed local signing identity for: $app_dir"
