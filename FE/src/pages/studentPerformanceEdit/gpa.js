import {
  Button,
  Card,
  Col,
  Input,
  InputNumber,
  Row,
  Space,
  TimePicker,
} from 'antd';
import TextArea from 'antd/es/input/TextArea';
import Dragger from 'antd/es/upload/Dragger';
import React, { useEffect, useState } from 'react';
import { InboxOutlined } from '@ant-design/icons';
import Swal from 'sweetalert2';
import { useDispatch, useSelector } from 'react-redux';
import { intializeStudDetails, submitGPAData } from '../../redux/action';
import jwt from 'jwt-decode';
import { Footer } from 'antd/lib/layout/layout';

const GpaEdit = ({ score }) => {
  const profileData = useSelector(
    (state) => state?.studentReducer?.profileData
  );
  const siginStatusCode = useSelector(
    (state) => state?.commonReducer?.siginStatusCode
  );
  const tokenDecoded =
    siginStatusCode === 200 ? jwt(localStorage.getItem('token')) : '';
  const dispatch = useDispatch();
  const [name, setName] = useState('');
  const [fileVal, setFile] = useState([]);

  const gpaStatus = useSelector(
    (state) => state?.studentDetailReducer?.gpaStatus
  );
  useEffect(() => {
    setName((profileData.gpas[0]?.value).toString() || '');
    intializeStudDetails(dispatch);
  }, []);

  useEffect(() => {
    if (gpaStatus === true) {
      Swal.fire({
        title: `GPA has been successfully saved`,
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
      setName(name);
      setFile([]);
      intializeStudDetails(dispatch);
    } else if (gpaStatus === false) {
      Swal.fire({
        title: `Error While Saving GPA details saved`,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });

      setFile([]);
      intializeStudDetails(dispatch);
    }
  }, [gpaStatus]);

  const handleFile = ({ fileList }) => {
    console.log(fileList);
    setFile([...fileList]);
  };

  const submitGPA = () => {
    if (!name || fileVal.length === 0) {
      Swal.fire({
        title: `Please enter required Fields to continue`,
        // text: "You won't be able to revert this!",
        icon: 'warning',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
    } else {
      let formData = new FormData();
      formData.append('file', fileVal[0]?.originFileObj);
      formData.append('name', name);
      formData.append('type', 'GPA');
      submitGPAData(formData, dispatch);
    }
  };

  const dummyRequest = ({ file, onSuccess }) => {
    setTimeout(() => {
      onSuccess('ok');
    }, 0);
  };
  {
    console.log(name, 'name?>>>>>');
  }
  return (
    <Row gutter={[24, 0]}>
      <Col xs={24} className="mb-24">
        <Card
            style={{background:"#EDF3FC"}}
          className="header-solid h-full ant-card-p-0"
          title={
            <>
              <Row
                gutter={[24, 0]}
                className="ant-row-flex ant-row-flex-middle"
              >
                <Col xs={24} md={12}>
                  <table>
                    <tbody>
                      <tr>
                        <td>
                          <div style={{ width: '50px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            GPA
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '90px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '600px' }}>
                          <InputNumber
                            style={{
                              width: 200,
                            }}
                            min="2"
                            max="4"
                            step="0.001"
                            onChange={(value) => setName(value)}
                            value={name}
                            stringMode
                          />
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <div style={{ height: '20px' }}></div>

                  <div style={{ height: '20px' }}></div>
                  <table>
                    <tbody>
                      <tr>
                        <td>
                          <div style={{ width: '50px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            Select a File
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '40px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '600px' }}>
                          <Dragger
                            fileList={fileVal}
                            maxCount={1}
                            multiple={false}
                            onChange={handleFile}
                            customRequest={dummyRequest}
                            //   onChange={handleFile}
                          >
                            <p className="ant-upload-drag-icon">
                              <InboxOutlined />
                            </p>
                            <p
                              className="ant-upload-text"
                              style={{
                                color: '#070C83',
                                fontWeight: 600,
                                fontSize: '10px',
                              }}
                            >
                              Click or drag file to this area to upload
                            </p>
                          </Dragger>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <div style={{ marginTop: '60px', marginLeft: '80px' }}>
                    <table>
                      <tbody>
                        <tr>
                          <td>
                            <div style={{ width: '50px' }}></div>
                          </td>
                          <td style={{ textAlign: 'left' }}>
                            <Button
                              onClick={submitGPA}
                              style={{
                                background:
                                  'linear-gradient(to right, #000AF3, #69D0F0)', // Replace with your desired gradient colors
                                border: 'none',
                                color: 'white',
                                width: '150px',
                              }}
                            >
                              Submit
                            </Button>
                          </td>
                          <td>
                            <div style={{ width: '20px' }}></div>
                          </td>

                          
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </Col>
              </Row>
            </>
          }
        ></Card>
      </Col>
      <Footer />
    </Row>
  );
};

export default GpaEdit;
