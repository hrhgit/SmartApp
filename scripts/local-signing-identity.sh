#!/bin/zsh
set -euo pipefail

identity_name="SmartApp Local Release"
keychain="$(security default-keychain -d user)"
keychain="${keychain#*\"}"
keychain="${keychain%%\"*}"

if [[ -z "$keychain" || ! -e "$keychain" ]]; then
  print -u2 "Could not locate the user's default keychain."
  exit 1
fi

matches=()
while IFS= read -r match; do
  [[ -n "$match" ]] && matches+=("$match")
done < <(security find-identity -v -p codesigning "$keychain" 2>/dev/null | /usr/bin/awk -v name="$identity_name" '
  $2 ~ /^[[:xdigit:]]+$/ && length($2) == 40 && index($0, "\"" name "\"") {
    print $2
  }
')

if (( ${#matches} == 0 )); then
  print -u2 "No valid \"${identity_name}\" signing identity was found."
  print -u2 "Run ./scripts/create-local-signing-identity.sh once before building SmartApp."
  exit 1
fi

if (( ${#matches} != 1 )); then
  print -u2 "Found more than one valid \"${identity_name}\" signing identity. Refusing to choose one."
  exit 1
fi

print -r -- "$matches[1]"
