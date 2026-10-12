import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {randomUUID} from 'node:crypto';
import {snapshotOf, foregroundKey, locationOnly} from '../../BrowserExtension/state.js';

const fixture = () => [{id:1,type:'normal',focused:true,state:'normal',tabs:[
  {id:11,index:0,active:true,url:'https://example.com/a?q=one#part'},
  {id:12,index:1,active:false,url:'https://example.com/background',pinned:true}]}];
test('state includes only the focused window active tab and full URL', () => {
  const s = snapshotOf(fixture());
  assert.equal(s.windows.length, 1);
  assert.equal(s.windows[0].tabs.length, 1);
  assert.equal(s.windows[0].activeID, '11');
  assert.equal(JSON.stringify(s).includes('background'), false);
  assert.equal(foregroundKey(s), '1:11:https://example.com/a?q=one#part');
});
test('exclude private, nonnormal windows, internal URLs, credentials; no title/body', () => {
  const input = fixture(); input[0].tabs[0].title='DO NOT COPY';
  input.push({...input[0],id:2,incognito:true}); input.push({...input[0],id:3,type:'popup'});
  const s = snapshotOf(input);
  assert.equal(s.windows.length, 1); assert.ok(!JSON.stringify(s).includes('DO NOT COPY'));
  assert.equal(locationOnly('chrome://settings'), '');
  assert.equal(locationOnly('https://user:password@example.com/a?q=1#b'),'https://example.com/a?q=1#b');
  input[0].state='minimized'; assert.equal(foregroundKey(snapshotOf(input)), null);
});

function harness(storage = {}) {
  const event = () => {const callbacks=[]; return {addListener:f=>callbacks.push(f), fire:(...args)=>callbacks.forEach(f=>f(...args))};};
  let windows=fixture(), queries=0;
  const messages=[], controls=[], timers=new Map(); let nextTimer=0;
  const port={onMessage:event(),onDisconnect:event(),postMessage:m=>(m.type === 'capabilities' ? controls : messages).push(m),disconnect(){this.onDisconnect.fire();}};
  const names=['onActivated','onCreated','onRemoved','onUpdated','onMoved','onAttached','onDetached','onReplaced'];
  const chrome={
    storage:{local:{get:async()=>({...storage}),set:async value=>Object.assign(storage,value)}},
    tabs:{...Object.fromEntries(names.map(n=>[n,event()])),query:async options=>{
      assert.equal(options.active,true); assert.equal(options.windowId,windows[0].id);
      return structuredClone(windows[0].tabs.filter(t=>t.active));
    }},
    windows:{...Object.fromEntries(['onFocusChanged','onCreated','onRemoved','onBoundsChanged'].map(n=>[n,event()])),getLastFocused:async options=>{
      assert.equal(options.populate,false); queries++;
      const window=structuredClone(windows[0]); delete window.tabs; return window;
    }},
    runtime:{id:'fixture',connectNative:()=>port,onMessage:event(),onStartup:event(),onInstalled:event()},
    action:{setBadgeText(){},setTitle(){}}, alarms:{create(){},onAlarm:event()},
    webNavigation:Object.fromEntries(['onCommitted','onHistoryStateUpdated','onReferenceFragmentUpdated'].map(n=>[n,event()]))
  };
  const context=vm.createContext({chrome,snapshotOf,foregroundKey,locationOnly,Date,Promise,crypto:{randomUUID},
    setTimeout:f=>{timers.set(++nextTimer,f);return nextTimer;}, clearTimeout:id=>timers.delete(id)});
  vm.runInContext(fs.readFileSync(new URL('../../BrowserExtension/background.js',import.meta.url),'utf8').replace(/^import .*;\n/,''),context);
  return {chrome,port,messages,controls,timers,setWindows:value=>windows=value,queries:()=>queries};
}
async function drain() {for(let i=0;i<40;i++) await Promise.resolve();}
test('worker does not collect before start or after pause and sends fresh state on resume',async()=>{
  const h=harness(); h.chrome.tabs.onActivated.fire(); await drain(); assert.equal(h.queries(),0);
  h.port.onMessage.fire({type:'capture',enabled:true,foreground:true,requestID:'start'}); await drain();
  assert.equal(h.messages[0].requestID,'start'); assert.equal(h.messages[0].reason,'initial');
  h.port.onMessage.fire({type:'capture',enabled:false,foreground:false}); const count=h.queries();
  h.chrome.tabs.onUpdated.fire(11,{url:'https://example.com/private-period'}); await drain(); assert.equal(h.queries(),count);
  const w=fixture(); w[0].tabs[0].url='https://example.com/resumed'; h.setWindows(w);
  h.port.onMessage.fire({type:'capture',enabled:true,foreground:true,requestID:'resume'}); await drain();
  assert.equal(h.messages.at(-1).snapshot.windows[0].tabs[0].url,'https://example.com/resumed');
  assert.ok(!JSON.stringify(h.messages).includes('private-period'));
});
test('SPA and fragment are top-frame-only; background navigation produces no foreground record',async()=>{
  const h=harness(); h.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  const before=h.messages.length; h.chrome.webNavigation.onHistoryStateUpdated.fire({frameId:2}); await drain();
  assert.equal(h.messages.length,before);
  const w=fixture(); w[0].tabs[1].url='https://example.com/spa#route'; h.setWindows(w);
  h.chrome.webNavigation.onHistoryStateUpdated.fire({frameId:0}); await drain();
  assert.equal(h.messages.length,before);
});
test('settle is a one-shot; changed location on delayed read is not falsely settled',async()=>{
  const h=harness(); h.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  assert.equal(h.timers.size,1);
  const w=fixture(); w[0].tabs[0].url='https://example.com/changed'; h.setWindows(w);
  const callback=[...h.timers.values()][0]; h.timers.clear(); callback(); await drain();
  assert.equal(h.messages.at(-1).reason,'tab_updated');
  const next=[...h.timers.values()][0]; h.timers.clear(); next(); await drain();
  assert.equal(h.messages.at(-1).reason,'settled'); assert.equal(h.timers.size,0);
});
test('pause invalidates an in-flight snapshot',async()=>{
  const h=harness(); h.port.onMessage.fire({type:'capture',enabled:true,foreground:true});
  h.port.onMessage.fire({type:'capture',enabled:false,foreground:false}); await drain(); assert.equal(h.messages.length,0);
});
test('profile identity survives worker restart but differs across profiles',async()=>{
  const storage={};
  const first=harness(storage); first.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  const restarted=harness(storage); restarted.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  const another=harness(); another.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  assert.ok(first.messages[0].profileID);
  assert.equal(first.messages[0].profileID,restarted.messages[0].profileID);
  assert.notEqual(first.messages[0].profileID,another.messages[0].profileID);
});
test('corrupt source identity stops capture without silently inventing a replacement',async()=>{
  const storage={smartappProfileID:'corrupt'};
  const h=harness(storage); h.port.onMessage.fire({type:'capture',enabled:true,foreground:true}); await drain();
  assert.equal(h.messages.length,0); assert.equal(storage.smartappProfileID,'corrupt');
});
test('explicit inspection returns only one active normal page and does not create a collection event',async()=>{
  const h=harness(); assert.equal(h.controls[0].foregroundInspection,true);
  h.port.onMessage.fire({type:'inspect_foreground',foreground:true,requestID:'lookup'}); await drain();
  assert.equal(h.messages.length,1); assert.equal(h.messages[0].type,'foreground_inspection');
  assert.equal(h.messages[0].message.snapshot.windows[0].tabs.length,1);
  assert.ok(!JSON.stringify(h.messages).includes('background'));
  h.chrome.tabs.onUpdated.fire(11,{url:'changed'}); await drain(); assert.equal(h.messages.length,1);
  const w=fixture(); w[0].incognito=true; h.setWindows(w);
  h.port.onMessage.fire({type:'inspect_foreground',foreground:true,requestID:'private'}); await drain();
  assert.equal(h.messages.at(-1).message.snapshot.windows.length,0);
  const count=h.queries();
  h.port.onMessage.fire({type:'inspect_foreground',foreground:false,requestID:'background'}); await drain();
  assert.equal(h.queries(),count);
});
