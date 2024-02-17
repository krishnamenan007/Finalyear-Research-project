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

const Gpa = () => {
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
    intializeStudDetails(dispatch);
    if (tokenDecoded && tokenDecoded?.sub?.is_gpa_submitted) {
      Swal.fire({
        title: `You have already subitted your GPA,You can proceed by updating `,
        // text: "You won't be able to revert this!",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
    }
  }, []);

  useEffect(() => {
    if (gpaStatus === true) {
      Swal.fire({
        title: `Details has been successfully saved`,
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
      setName('');
      setFile([]);
      intializeStudDetails(dispatch);
    } else if (gpaStatus === false) {
      Swal.fire({
        title: `Error While Saving details`,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
      setName('');
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
  return (
    <Row gutter={[24, 0]}>
      <Col xs={24} className="mb-24" style={{ width: '100%' }}>
        <Card
            style={{width: '90%',background:"#EDF3FC"}}
          className="header-solid h-full ant-card-p-0"
          title={
              <Row
                  style={{width: '100%'}}
                gutter={[24, 0]}
                className="ant-row-flex ant-row-flex-middle"
              >
                <Col style={{width: '100%'}}>
                  <div style={{ width: '100%' ,display:'flex',justifyContent:'center',alignItems:'center',gap:'20px'}}>
                          <span
                            style={{

                              width: '30%',
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            GPA Value
                          </span>
                          {/* <Input
                            value={name}
                            acc
                            onChange={(e) => setName(e.target.value)}
                            // value={des}
                            //   style={{ height: '150px' }}
                            // onChange={(e) => setDesc(e.target.value)}
                          /> */}

                          <InputNumber
                            style={{
                              width: '40%',
                            }}
                            min="2"
                            max="4"
                            step="0.001"
                            onChange={(value) => setName(value)}
                            stringMode
                          />
                  </div>
                  <div style={{display:'flex',justifyContent:'center',alignItems:'center',marginTop:'20px',width:'100%',gap:'20px'}}>
                    <span
                        style={{
                            width: '30%',
                          color: '#070C83',
                          fontWeight: 600,
                          fontSize: '15px',
                        }}
                    >
                            Select a File
                          </span>
                    <Dragger
                        style={{
                            width: '100%',
                        }}
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
                  </div>


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
                                fontSize: '20px',
                                background:
                                  'linear-gradient(to bottom, #000AF3, #69D0F0)', // Replace with your desired gradient colors
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

                          <td style={{ textAlign: 'left', width: '600px' }}>
                            <Button
                              onClick={() => {
                                setName('');
                                setFile([]);
                              }}
                              style={{
                                fontSize: '20px',
                                background:
                                  'linear-gradient(to bottom, #000AF3, #69D0F0)', // Replace with your desired gradient colors
                                border: 'none',
                                color: 'white',
                                width: '150px',
                              }}
                            >
                              Reset
                            </Button>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </Col>
              </Row>
          }
        ></Card>
      </Col>
    </Row>
  );
};

export default Gpa;
