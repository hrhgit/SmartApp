import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const home = os.homedir();
const support = path.join(home, 'Library/Application Support');
const app = path.join(home, 'Applications/SmartApp.app');
const host = path.join(app, 'Contents/MacOS/smartapp-browser-host');
if (!fs.existsSync(host)) throw new Error('Install the new SmartApp build before registering the browser host.');
const manifest = JSON.parse(fs.readFileSync(path.join(project, 'BrowserExtension/manifest.json')));
const id = [...crypto.createHash('sha256').update(Buffer.from(manifest.key, 'base64')).digest().subarray(0,16)]
  .map(b => String.fromCharCode(97+(b>>4),97+(b&15))).join('');
const root = path.join(support, 'SmartApp/integrations/browser-v1');
fs.mkdirSync(root, {recursive:true, mode:0o700});
const extension = path.join(root, 'extension');
if (fs.existsSync(extension)) fs.cpSync(extension, path.join(root, `extension-backup-${Date.now()}`), {recursive:true});
fs.cpSync(path.join(project, 'BrowserExtension'), extension, {recursive:true});
const quote = s => "'" + s.replaceAll("'", "'\\''") + "'";
const browsers = [['chrome','com.google.Chrome','Google/Chrome'], ['edge','com.microsoft.edgemac','Microsoft Edge'], ['brave','com.brave.Browser','BraveSoftware/Brave-Browser']];
for (const [name,bundle,folder] of browsers) {
  const wrapper = path.join(root, `host-${name}`);
  fs.writeFileSync(wrapper, `#!/bin/sh\nexec ${quote(host)} ${quote(bundle)} "$@"\n`, {mode:0o700});
  const destination = path.join(support, folder, 'NativeMessagingHosts');
  fs.mkdirSync(destination, {recursive:true, mode:0o700});
  const file = path.join(destination, 'local.herh.smartapp.browser.json');
  if (fs.existsSync(file)) fs.copyFileSync(file, `${file}.backup-${Date.now()}`);
  fs.writeFileSync(file, JSON.stringify({name:'local.herh.smartapp.browser', description:'SmartApp local browser bridge',
    path:wrapper, type:'stdio', allowed_origins:[`chrome-extension://${id}/`]}, null, 2), {mode:0o600});
}
console.log(`Extension ID: ${id}\nLoad unpacked extension from:\n${extension}\nHost registrations installed for Chrome, Edge and Brave. No browser preferences were changed.`);
