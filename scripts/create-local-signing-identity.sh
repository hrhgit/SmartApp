#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h:h}"
identity_name="SmartApp Local Release"
bundle_identifier="local.herh.SmartApp"
keychain="$(security default-keychain -d user)"
keychain="${keychain#*\"}"
keychain="${keychain%%\"*}"

if [[ -z "$keychain" || ! -e "$keychain" ]]; then
  print -u2 "Could not locate the user's default keychain."
  exit 1
fi

if existing_identity="$($project_dir/scripts/local-signing-identity.sh 2>/dev/null)"; then
  print "Using existing local SmartApp signing identity: $existing_identity"
  exit 0
fi

if security find-certificate -a -c "$identity_name" -Z "$keychain" 2>/dev/null | /usr/bin/grep -q 'SHA-1 hash:'; then
  print -u2 "A certificate named \"${identity_name}\" already exists but is not a valid signing identity."
  print -u2 "It is not replaced automatically because doing so could invalidate existing privacy authorizations."
  exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
  print -u2 "OpenSSL is required once to create the local signing identity."
  exit 1
fi

umask 077
temporary_dir="$(mktemp -d)"
private_key="$temporary_dir/private-key.pem"
certificate="$temporary_dir/certificate.pem"
identity_archive="$temporary_dir/identity.p12"
archive_password="$(openssl rand -hex 32)"

cleanup() {
  /bin/unlink "$private_key" >/dev/null 2>&1 || true
  /bin/unlink "$certificate" >/dev/null 2>&1 || true
  /bin/unlink "$identity_archive" >/dev/null 2>&1 || true
  /bin/rmdir "$temporary_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT

openssl req -x509 -newkey rsa:3072 -sha256 -days 3650 -nodes \
  -keyout "$private_key" \
  -out "$certificate" \
  -subj "/CN=${identity_name}/OU=${bundle_identifier}/O=SmartApp" \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,digitalSignature,keyCertSign' \
  -addext 'extendedKeyUsage=critical,codeSigning' \
  -addext 'subjectKeyIdentifier=hash' \
  -addext 'authorityKeyIdentifier=keyid:always' >/dev/null 2>&1

openssl pkcs12 -export \
  -inkey "$private_key" \
  -in "$certificate" \
  -out "$identity_archive" \
  -passout "pass:$archive_password"

security import "$identity_archive" \
  -k "$keychain" \
  -f pkcs12 \
  -P "$archive_password" \
  -T /usr/bin/codesign >/dev/null
security add-trusted-cert \
  -r trustRoot \
  -p codeSign \
  -k "$keychain" \
  "$certificate" >/dev/null

created_identity="$($project_dir/scripts/local-signing-identity.sh)"
print "Created local SmartApp signing identity: $created_identity"
print "It is valid for ten years and trusted only for code signing in this user's login keychain."
