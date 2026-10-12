#!/bin/zsh
set -euo pipefail
project_dir="${0:A:h:h}"
cd "$project_dir"
configuration="${1:-release}"
if [[ "$configuration" != "release" && "$configuration" != "debug" ]]; then
  print -u2 "Usage: scripts/build-app.sh [release|debug]"
  exit 1
fi
swift build -c "$configuration"
binary_dir="$(swift build -c "$configuration" --show-bin-path)"
app_dir="$project_dir/build/SmartApp.app"
mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"
cp "$binary_dir/SmartApp" "$app_dir/Contents/MacOS/SmartApp"
cp "$binary_dir/smartapp-browser-host" "$app_dir/Contents/MacOS/smartapp-browser-host"
cp "$binary_dir/smartapp-directory-host" "$app_dir/Contents/MacOS/smartapp-directory-host"
cp "$project_dir/Resources/Info.plist" "$app_dir/Contents/Info.plist"
if [[ ! -f "$project_dir/Resources/AppIcon.icns" || ! -f "$project_dir/Resources/MenuBarIcon.png" ]]; then
  swift "$project_dir/scripts/make-icon.swift"
fi
cp "$project_dir/Resources/AppIcon.icns" "$app_dir/Contents/Resources/AppIcon.icns"
cp "$project_dir/Resources/MenuBarIcon.png" "$app_dir/Contents/Resources/MenuBarIcon.png"
cp "$project_dir/Resources/KevSmartAppServer.py" "$app_dir/Contents/Resources/KevSmartAppServer.py"
cp "$binary_dir/smartapp-cli" "$project_dir/build/smartapp-cli"
signing_identity="$($project_dir/scripts/local-signing-identity.sh)"
codesign --force --sign "$signing_identity" --identifier local.herh.SmartApp.BrowserHost "$app_dir/Contents/MacOS/smartapp-browser-host"
codesign --force --sign "$signing_identity" --identifier local.herh.SmartApp.DirectoryHost "$app_dir/Contents/MacOS/smartapp-directory-host"
codesign --force --sign "$signing_identity" --identifier local.herh.SmartApp "$app_dir"
"$project_dir/scripts/verify-app-signature.sh" "$app_dir"
print "Built: $app_dir"
