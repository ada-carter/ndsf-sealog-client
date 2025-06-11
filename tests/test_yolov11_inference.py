import io
import csv
import unittest
from pathlib import Path
from unittest import mock

import tools.yolov11_inference as yi

class TestYoloV11Inference(unittest.TestCase):
    def setUp(self):
        self.fake_img = Path('fake.jpg')

    @mock.patch('tools.yolov11_inference.YOLO')
    def test_run_inference(self, mock_yolo):
        mock_model = mock.Mock()
        mock_yolo.return_value = mock_model
        result_box = [1,2,3,4,0.9,0]
        fake_results = mock.Mock()
        fake_results.boxes.data.tolist.return_value = [result_box]
        mock_model.return_value = [fake_results]

        preds = yi.run_inference([self.fake_img], 'weights.pt', conf=0.5, iou=0.4)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]['class'], 0)
        self.assertAlmostEqual(preds[0]['score'], 0.9)

    def test_write_csv(self):
        preds = [{'image': 'img.jpg', 'class': 1, 'score': 0.8, 'x1':0,'y1':0,'x2':1,'y2':1}]
        m = mock.mock_open()
        with mock.patch('builtins.open', m):
            yi.write_csv(preds, 'out.csv')
        handle = m()
        written = ''.join(call.args[0] for call in handle.write.call_args_list)
        reader = csv.DictReader(written.splitlines())
        rows = list(reader)
        self.assertEqual(rows[0]['class'], '1')

    @mock.patch('tools.yolov11_inference.YOLO', None)
    def test_no_ultralytics(self):
        with self.assertRaises(RuntimeError):
            yi.run_inference([self.fake_img], 'weights.pt')

if __name__ == '__main__':
    unittest.main()
