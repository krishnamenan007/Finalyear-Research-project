import { Button, Card, Col, Input, Row, Select, Tabs } from 'antd';
import { useEffect, useState } from 'react';
import { intializeStudDetails, submitGPAData } from '../../redux/action';
import { useDispatch, useSelector } from 'react-redux';
import Swal from 'sweetalert2';
import Dragger from 'antd/lib/upload/Dragger';
import { InboxOutlined } from '@ant-design/icons';

const SelfLearning = () => {
  const dispatch = useDispatch();
  const [name, setName] = useState('');
  const [fileVal, setFile] = useState([]);
  const [courseLevel, setLevel] = useState('beginner');
  const [number, setNumber] = useState('');

  const gpaStatus = useSelector(
    (state) => state?.studentDetailReducer?.gpaStatus
  );
  useEffect(() => {
    intializeStudDetails(dispatch);
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
      setNumber();
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
      setNumber('');
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
      formData.append('course_name', name);
      formData.append('certificate_no', number);
      formData.append('course_level', courseLevel);

      formData.append('type', 'SELF_LEARNING');
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
                          <div style={{ width: '30px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            Course/Project Name
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '65px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '600px' }}>
                          <Input
                            style={{ width: '200px' }}
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            // value={des}
                            //   style={{ height: '150px' }}
                            // onChange={(e) => setDesc(e.target.value)}
                          />
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <div style={{ height: '20px' }}></div>

                  <table>
                    <tbody>
                      <tr>
                        <td>
                          <div style={{ width: '30px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            Certificate Number
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '65px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '600px' }}>
                          <Input
                            style={{ width: '200px' }}
                            value={number}
                            onChange={(e) => setNumber(e.target.value)}
                            // value={des}
                            //   style={{ height: '150px' }}
                            // onChange={(e) => setDesc(e.target.value)}
                          />
                        </td>
                      </tr>
                    </tbody>
                  </table>


                  <table>
                    <tbody>
                      <tr>
                        <td>
                          <div style={{ width: '30px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                            }}
                          >
                            Course/Project Level
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '75px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '600px' }}>
                          <Select
                            style={{ width: '200px' }}
                            defaultValue="beginner"
                            onChange={(e) => setLevel(e)}
                            options={[
                              { value: 'beginner', label: 'beginner' },
                              { value: 'intermediate', label: 'intermediate' },
                              { value: 'expert', label: 'expert' },
                            ]}
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
                          <div style={{ width: '30px' }}></div>
                        </td>
                        <td style={{ textAlign: 'left' }}>
                          <span
                            style={{
                              color: '#070C83',
                              fontWeight: 600,
                              fontSize: '15px',
                              marginLeft: '5px',
                            }}
                          >
                            Select a File
                          </span>
                        </td>
                        <td>
                          <div style={{ width: '130px' }}></div>
                        </td>

                        <td style={{ textAlign: 'left', width: '400px' }}>
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
                  <div style={{ marginTop: '30px', marginLeft: '100px' }}>
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

                          <td style={{ textAlign: 'left', width: '600px' }}>
                            <Button
                              onClick={() => {
                                setName('');
                                setFile([]);
                              }}
                              style={{
                                background:
                                  'linear-gradient(to right, #000AF3, #69D0F0)', // Replace with your desired gradient colors
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
            </>
          }
        ></Card>
      </Col>
    </Row>
  );
};

export default SelfLearning;
