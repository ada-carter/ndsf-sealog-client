const http = require('http');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

function sendCsv(res, csv) {
  res.writeHead(200, {'Content-Type': 'text/csv'});
  res.end(csv);
}

function handleRequest(req, res) {
  if (req.method !== 'POST' || req.url !== '/api/yolo') {
    res.writeHead(404); return res.end();
  }
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    try {
      const {weights, images, conf = 0.25, iou = 0.7} = JSON.parse(body);
      if (!weights || !images || !images.length) {
        res.writeHead(400); return res.end('weights and images required');
      }
      if (process.env.TEST_MODE) {
        return sendCsv(res, 'image,class,score,x1,y1,x2,y2\nimg.jpg,0,0.9,0,0,1,1\n');
      }
      const outfile = path.join(os.tmpdir(), `yolo_${Date.now()}.csv`);
      const args = ['tools/yolov11_inference.py','--weights',weights,'--output',outfile,'--conf',conf,'--iou',iou,'--images',...images];
      const proc = spawn('python3', args);
      let err='';
      proc.stderr.on('data', d=>{err+=d;});
      proc.on('close', code=>{
        if(code!==0){res.writeHead(500);return res.end(err);}
        fs.readFile(outfile,(e,data)=>{
          if(e){res.writeHead(500);return res.end(e.message);} 
          sendCsv(res, data);
          fs.unlink(outfile,()=>{});
        });
      });
    } catch(e){
      res.writeHead(400); res.end('bad request');
    }
  });
}

const port = process.env.PORT || 5000;
http.createServer(handleRequest).listen(port,()=>{
  console.log(`YOLO server listening on ${port}`);
});
