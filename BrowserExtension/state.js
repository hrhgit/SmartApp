export function snapshotOf(windows) {
  // Only the current focused normal window and its active tab may cross the
  // native boundary. Background windows/tabs are never placed in a snapshot.
  const w = windows.find(candidate => candidate.type === 'normal' && !candidate.incognito &&
    !!candidate.focused && candidate.state !== 'minimized');
  if (!w) return {windows: []};
  const active = (w.tabs || []).find(tab => tab.active && !tab.incognito);
  return {windows: [{
    id: String(w.id), front: true, activeID: String(active?.id ?? 0), minimized: false,
    tabs: active ? [{id: String(active.id), url: locationOnly(active.url), index: active.index, pinned: !!active.pinned}] : []
  }]};
}
export function locationOnly(value) {
  try { const u = new URL(value); if (!['http:', 'https:'].includes(u.protocol)) return '';
    u.username = ''; u.password = ''; return u.href;
  } catch { return ''; }
}
export function foregroundKey(snapshot) {
  const w = snapshot.windows.find(w => w.front);
  const t = w?.tabs.find(t => t.id === w.activeID);
  return t?.url ? `${w.id}:${t.id}:${t.url}` : null;
}
