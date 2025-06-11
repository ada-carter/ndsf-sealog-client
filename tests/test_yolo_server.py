import json
import unittest
import subprocess
import time
import urllib.request

class TestYoloServer(unittest.TestCase):
    def setUp(self):
        node = subprocess.getoutput('which node').strip() or 'node'
        self.proc = subprocess.Popen([node,'tools/yolo_server.js'], env={**dict(PORT='5051', TEST_MODE='1')})
        time.sleep(1)

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait()

    def test_endpoint(self):
        data = json.dumps({'weights':'w.pt','images':['i.jpg']}).encode()
        req = urllib.request.Request('http://localhost:5051/api/yolo', data=data, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
        self.assertIn('image,class', body)

if __name__ == '__main__':
    unittest.main()
