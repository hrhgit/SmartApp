#!/bin/zsh
set -euo pipefail
project_dir="${0:A:h:h}"
if [[ "${1:-}" != "" && "${1:-}" != "--collect" ]]; then
  print -u2 "Usage: scripts/install-app.sh [--collect]"
  exit 1
fi
"$project_dir/scripts/build-app.sh" release
target_dir="$HOME/Applications/SmartApp.app"
if /usr/bin/pgrep -x SmartApp >/dev/null; then
  print -u2 "Please quit SmartApp from its menu bar before installing an update."
  exit 1
fi
mkdir -p "$HOME/Applications"
if [[ -L "$target_dir" || ( -e "$target_dir" && ! -d "$target_dir" ) ]]; then
  print -u2 "Refusing to replace a non-directory or symbolic-link install target: $target_dir"
  exit 1
fi
if [[ -e "$target_dir" ]]; then
  backup_dir="$HOME/Applications/SmartApp.backup.$(date +%Y%m%d-%H%M%S).app"
  ditto "$target_dir" "$backup_dir"
  print "Previous app backed up to: $backup_dir"
  rsync -aE --delete "$project_dir/build/SmartApp.app/" "$target_dir/"
else
  ditto "$project_dir/build/SmartApp.app" "$target_dir"
fi
"$project_dir/scripts/verify-app-signature.sh" "$target_dir"
if [[ "${1:-}" == "--collect" ]]; then
  open -a "$target_dir" --args --collect --background
else
  open "$target_dir"
fi
print "Installed: $target_dir"
