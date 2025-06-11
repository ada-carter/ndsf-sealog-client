import React, { Component } from 'react';
import { connectModal } from 'redux-modal';
import { Modal, Button, Form, Row, Col } from 'react-bootstrap';
import axios from 'axios';
import FileDownload from 'js-file-download';

class YoloInferenceModal extends Component {
  constructor(props) {
    super(props);
    this.state = {weights: '', images: '', conf: 0.25, iou: 0.7};
  }

  runInference = async () => {
    try {
      const {weights, images, conf, iou} = this.state;
      const payload = {weights, images: images.split(',').map(i => i.trim()).filter(Boolean), conf: parseFloat(conf), iou: parseFloat(iou)};
      const res = await axios.post('/api/yolo', payload, {responseType: 'blob'});
      FileDownload(res.data, 'predictions.csv');
    } catch (err) {
      alert('YOLO inference failed');
    }
  };

  render() {
    const {show, handleHide} = this.props;
    return (
      <Modal show={show} onHide={handleHide}>
        <Modal.Header closeButton>
          <Modal.Title>Run YOLOv11 Inference</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group>
              <Form.Label>Weights file (.pt)</Form.Label>
              <Form.Control type="text" value={this.state.weights} onChange={e=>this.setState({weights: e.target.value})} />
            </Form.Group>
            <Form.Group>
              <Form.Label>Image paths (comma separated)</Form.Label>
              <Form.Control type="text" value={this.state.images} onChange={e=>this.setState({images: e.target.value})} />
            </Form.Group>
            <Row>
              <Col>
                <Form.Group>
                  <Form.Label>Conf</Form.Label>
                  <Form.Control type="number" step="0.01" value={this.state.conf} onChange={e=>this.setState({conf: e.target.value})} />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>IoU</Form.Label>
                  <Form.Control type="number" step="0.01" value={this.state.iou} onChange={e=>this.setState({iou: e.target.value})} />
                </Form.Group>
              </Col>
            </Row>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button size="sm" variant="secondary" onClick={handleHide}>Close</Button>
          <Button size="sm" onClick={this.runInference}>Run</Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default connectModal({ name: 'yoloInference' })(YoloInferenceModal);
