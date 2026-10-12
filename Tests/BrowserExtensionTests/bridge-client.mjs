// Synthetic TCP peer for XCTest. Never reads the production endpoint.
import fs from 'node:fs';
import net from 'node:net';
const config=JSON.parse(fs.readFileSync(process.argv[2]));
const mode=process.argv[3],bad=mode==='bad',inspection=mode==='inspection'||mode==='cancel-inspection';
const socket=net.connect(config.port,'127.0.0.1');
const frame=value=>{const body=Buffer.from(JSON.stringify(value)),header=Buffer.alloc(4);header.writeUInt32LE(body.length);return Buffer.concat([header,body]);};
socket.on('connect',()=>{
  socket.write(frame({type:'hello',token:bad?'wrong':config.token,bundle:'com.google.Chrome'}));
  if(inspection) socket.write(frame({type:'capabilities',foregroundInspection:true}));
});
let buffer=Buffer.alloc(0),received=false;
socket.on('data',data=>{
  buffer=Buffer.concat([buffer,data]);
  while(buffer.length>=4 && buffer.length>=4+buffer.readUInt32LE()) {
    const size=buffer.readUInt32LE(),control=JSON.parse(buffer.subarray(4,4+size));buffer=buffer.subarray(4+size);
    if(control.type==='inspect_foreground' && inspection) {
      received=true;
      if(mode==='inspection') {
        socket.write(frame({type:'foreground_inspection',requestID:control.requestID,
          message:{version:2,sequence:1,capturedAt:Date.now(),reason:'resync',snapshot:{windows:[{id:'9',front:true,minimized:false,activeID:'1',
            tabs:[{id:'1',url:'https://example.com/inspection',index:0,pinned:false}]}]}}}));
        socket.end();
      }
      continue;
    }
    if(control.type==='capture' && control.enabled && control.foreground) {
      received=true;
      const message={version:2,sequence:1,capturedAt:Date.now(),reason:'initial',requestID:control.requestID,
        snapshot:{windows:[{id:'9',front:true,minimized:false,activeID:'1',tabs:[{id:'1',url:'https://example.com/bridge-test',index:0,pinned:false}]}]}};
      // Deliberately split the frame, then repeat its sequence to test replay rejection.
      const bytes=frame(message); socket.write(bytes.subarray(0,3));socket.write(bytes.subarray(3));
      setTimeout(()=>socket.write(frame(message)),30);
    }
  }
});
socket.on('close',()=>{process.exit(bad||received?0:2);});
socket.on('error',()=>process.exit(3));
setTimeout(()=>process.exit(4),4000);
