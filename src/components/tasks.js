import React, { Component } from 'react';
import { compose } from 'redux';
import { connect } from 'react-redux';
import { Row, Col, Container, ListGroup } from 'react-bootstrap';
import ImportEventsModal from './import_events_modal';
import ImportAuxDataModal from './import_aux_data_modal';
import DataWipeModal from './data_wipe_modal';
import YoloInferenceModal from './yolo_inference_modal';
import * as mapDispatchToProps from '../actions';

const importEventsDescription = (<div><h5>Import Event Records</h5><p>Add new event data records from a JSON-formated file.</p></div>);

const importAuxDataDescription = (<div><h5>Import Aux Data Records</h5><p>Add new aux data records from a JSON-formated file.</p></div>);

const dataResetDescription = (<div><h5>Wipe Local Database</h5><p>Delete all existing events from the local database.</p></div>);

const yoloDescription = (<div><h5>Run YOLOv11 Detection</h5><p>Detect objects in images using a local weights file. Results will be downloaded as CSV.</p></div>);

class Tasks extends Component {

  constructor (props) {
    super(props);

    this.state = {
      description: "" 
    };
  }

  handleEventImport() {
    this.props.showModal('importEvents');
  }

  handleAuxDataImport() {
    this.props.showModal('importAuxData');
  }

  handleDataWipe() {
    this.props.showModal('dataWipe', { handleDelete: this.props.deleteAllEvents });
  }

  handleYoloInference() {
    this.props.showModal("yoloInference");
  }

  componentDidMount() {
  }

  renderTaskTable() {
    return (
      <ListGroup>
        <ListGroup.Item onMouseEnter={() => this.setState({ description: importEventsDescription })} onMouseLeave={() => this.setState({ description: "" })} onClick={ () => this.handleEventImport()}>Import Event Records</ListGroup.Item>
        <ListGroup.Item onMouseEnter={() => this.setState({ description: importAuxDataDescription })} onMouseLeave={() => this.setState({ description: "" })} onClick={ () => this.handleAuxDataImport()}>Import Aux Data Records</ListGroup.Item>
        <ListGroup.Item onMouseEnter={() => this.setState({ description: dataResetDescription })} onMouseLeave={() => this.setState({ description: "" })} onClick={ () => this.handleDataWipe()}>Wipe Local Database</ListGroup.Item>
        <ListGroup.Item onMouseEnter={() => this.setState({ description: yoloDescription })} onMouseLeave={() => this.setState({ description: "" })} onClick={() => this.handleYoloInference()}>Run YOLOv11 Detection</ListGroup.Item>
      </ListGroup>
    );
  }

  render() {
    if (!this.props.roles) {
      return (
        <div>Loading...</div>
      );
    }

    else if(this.props.roles.includes("admin")) {
      return (
        <div>
          <ImportEventsModal />
          <ImportAuxDataModal />
          <DataWipeModal />
          <YoloInferenceModal />
          <Row>
            <Col sm={5} md={{span:4, offset:1}} lg={{span:3, offset:2}}>
              {this.renderTaskTable()}
            </Col>
            <Col sm={7} md={6} lg={5}>
              <Container>
                <Row>
                  <Col>
                    {this.state.description}
                  </Col>
                </Row>
              </Container>
            </Col>
          </Row>
        </div>
      );

    } else {
      this.props.gotoHome();
    }
  }
}

function mapStateToProps(state) {

  return {
    roles: state.user.profile.roles,
  };
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
)(Tasks);