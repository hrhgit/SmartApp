import {snapshotOf, foregroundKey, locationOnly} from './state.js';

let port = null, enabled = false, foreground = false, sequence = 0, generation = 0;
let chain = Promise.resolve(), pending = 0, settleTimer, lastKey = null;
let status = '未连接 SmartApp';
// Random profile-local identity: reconnects must not be mistaken for another source.
const profileIdentity = chrome.storage.local.get('smartappProfileID').then(async saved => {
  if (saved.smartappProfileID !== undefined) {
    if (typeof saved.smartappProfileID !== 'string' || !/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(saved.smartappProfileID)) {
      throw new Error('Invalid local source identity');
    }
    return saved.smartappProfileID;
  }
  const id = crypto.randomUUID();
  await chrome.storage.local.set({smartappProfileID: id});
  return id;
}).catch(() => null);
const badge = (text, title) => { status = title; chrome.action.setBadgeText({text}); chrome.action.setTitle({title}); };
function lost(current) {
  if (port !== current) return;
  port = null; enabled = false; foreground = false; generation++;
  clearTimeout(settleTimer); lastKey = null; badge('!', 'SmartApp 未连接；点击重新连接');
  chrome.alarms.create('reconnect', {delayInMinutes: 0.5});
}
function disconnect(current) { lost(current); current.disconnect(); }

function connect() {
  if (port) return;
  const current = chrome.runtime.connectNative('local.herh.smartapp.browser');
  port = current; enabled = false; foreground = false; generation++; sequence = 0;
  badge('…', '正在连接 SmartApp');
  current.onDisconnect.addListener(() => {
    if (port !== current) return;
    void chrome.runtime.lastError; lost(current);
    // Connection retry only: no polling of browser state while disconnected.
  });
  current.onMessage.addListener(message => {
    if (port !== current) return;
    if (message.type === 'capture') {
      generation++; enabled = message.enabled === true; foreground = message.foreground === true; lastKey = null; clearTimeout(settleTimer);
      badge(enabled && foreground ? '' : 'Ⅱ', enabled && foreground ? 'SmartApp 已连接，前台页面采集中' : 'SmartApp 已连接，等待浏览器前台');
      if (enabled && foreground) capture('initial', message.requestID);
    } else if (message.type === 'resync' && enabled && foreground) {
      capture('resync', message.requestID);
    } else if (message.type === 'activate') {
      activate(message, current);
    } else if (message.type === 'inspect_foreground' && message.foreground === true) {
      inspectForeground(message, current);
    }
  });
  current.postMessage({type: 'capabilities', foregroundInspection: true});
}

async function foregroundSnapshot() {
  const window = await chrome.windows.getLastFocused({populate:false, windowTypes:['normal']});
  if (!window || !window.focused || window.type !== 'normal' || window.incognito || window.state === 'minimized') {
    return {windows: []};
  }
  // Ask for one active tab directly; never obtain the other tabs in this window.
  const tabs = await chrome.tabs.query({active:true, windowId:window.id});
  return snapshotOf([{...window, tabs}]);
}

async function inspectForeground(message, current) {
  try {
    const snapshot = await foregroundSnapshot();
    if (port !== current) return;
    current.postMessage({type:'foreground_inspection', requestID:message.requestID,
      message:{version:2, sequence:1, capturedAt:Date.now(), reason:'resync', snapshot}});
  } catch {
    if (port === current) current.postMessage({type:'foreground_inspection', requestID:message.requestID, error:true});
  }
}

async function activate(message, current) {
  let success = false;
  try {
    const tab = await chrome.tabs.get(Number(message.tabID));
    if (!tab.incognito && String(tab.windowId) === message.windowID &&
        locationOnly(tab.url) === message.url) {
      await chrome.tabs.update(tab.id, {active: true});
      await chrome.windows.update(tab.windowId, {focused: true}); success = true;
    }
  } catch { /* Closed, navigated, or inaccessible entrance: app reopens saved URL. */ }
  if (port === current) current.postMessage({type: 'activation_result', requestID: message.requestID, success});
}

function capture(reason, requestID = undefined) {
  if (!enabled || !foreground || !port) return;
  if (pending >= 32) { // Bound transient work; signal loss instead of inventing intermediate visits.
    disconnect(port); return;
  }
  const epoch = generation, current = port;
  pending++;
  chain = chain.then(async () => {
    if (epoch !== generation || !enabled || !foreground || port !== current) return;
    const snapshot = await foregroundSnapshot();
    const profileID = await profileIdentity;
    if (!profileID) throw new Error('Source identity unavailable');
    if (epoch !== generation || !enabled || !foreground || port !== current) return;
    const key = foregroundKey(snapshot);
    if (!['initial', 'resync', 'settled'].includes(reason) && key === lastKey) return;
    if (reason === 'settled' && key !== lastKey) reason = 'tab_updated';
    current.postMessage({version:2, sequence:++sequence, capturedAt:Date.now(), reason, requestID, profileID, snapshot});
    if (reason !== 'settled' && (key !== lastKey || ['initial','resync','tab_activated','window_focus'].includes(reason))) {
      clearTimeout(settleTimer); lastKey = key;
      if (key) settleTimer = setTimeout(() => { if (epoch === generation) capture('settled'); }, 1200);
    }
  }).catch(() => {
    if (port === current) disconnect(current);
  }).finally(() => pending--);
}

chrome.tabs.onActivated.addListener(() => capture('tab_activated'));
chrome.tabs.onCreated.addListener(() => capture('tab_created'));
chrome.tabs.onRemoved.addListener(() => capture('tab_removed'));
chrome.tabs.onUpdated.addListener((id, change) => {
  if (change.url !== undefined || change.status !== undefined || change.pinned !== undefined) capture('tab_updated');
});
chrome.tabs.onMoved.addListener(() => capture('tab_moved'));
chrome.tabs.onAttached.addListener(() => capture('tab_attached'));
chrome.tabs.onDetached.addListener(() => capture('tab_detached'));
chrome.tabs.onReplaced.addListener(() => capture('tab_replaced'));
chrome.windows.onFocusChanged.addListener(() => capture('window_focus'));
chrome.windows.onCreated.addListener(() => capture('window_created'));
chrome.windows.onRemoved.addListener(() => capture('window_removed'));
chrome.windows.onBoundsChanged.addListener(() => capture('window_bounds'));
for (const [event, reason] of [[chrome.webNavigation.onCommitted,'navigation'],
  [chrome.webNavigation.onHistoryStateUpdated,'history'], [chrome.webNavigation.onReferenceFragmentUpdated,'fragment']]) {
  event.addListener(details => { if (details.frameId === 0) capture(reason); });
}
chrome.alarms.onAlarm.addListener(a => { if (a.name === 'reconnect') connect(); });
chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (sender.id !== chrome.runtime.id) return;
  if (message.type === 'reconnect') { if (port) capture('resync'); else connect(); }
  reply({status, connected: !!port, capturing:enabled});
});
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
connect();
