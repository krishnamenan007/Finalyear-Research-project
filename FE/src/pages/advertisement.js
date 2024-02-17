import {
  Button,
  Card,
  Col,
  Input,
  Modal,
  Row,
  Space,
  Table,
  DatePicker,
  TimePicker,
} from 'antd';
import React, { useEffect, useState } from 'react';
import {
  createAD,
  deleteAd,
  getAllAds,
  initailizeAd,
  updateAd,
} from '../redux/action';
import Swal from 'sweetalert2';
import moment from 'moment';
import { useDispatch, useSelector } from 'react-redux';
import TextArea from 'antd/lib/input/TextArea';
import 'react-time-picker/dist/TimePicker.css';
import 'react-clock/dist/Clock.css';
import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';
dayjs.extend(customParseFormat);
const format = 'HH:mm';

// import DatePicker from 'react-datepicker';

const Advertisement = () => {
  const dispatch = useDispatch();
  const [modalStaus, setModal] = useState({
    status: false,
    type: '',
  });

  const [adDataVal, setAdData] = useState([]);

  const adData = useSelector((state) => state?.advertisementReducer?.adData);
  const getAdStatus = useSelector(
    (state) => state?.advertisementReducer?.getAdStatus
  );

  const adCreateStatus = useSelector(
    (state) => state?.advertisementReducer?.adCreateStatus
  );

  const updateAdStatus = useSelector(
    (state) => state?.advertisementReducer?.updateAdStatus
  );

  const deleteAdStatus = useSelector(
    (state) => state?.advertisementReducer?.deleteAdStatus
  );

  const [timeVal, setTime] = useState(null);
  const [venue, setVenue] = useState('');
  const [name, setName] = useState('');
  const [date, setDate] = useState(null);
  const [des, setDesc] = useState('');
  const [id, setId] = useState(null);

  useEffect(() => {
    initailizeAd(dispatch);
    getAllAds(dispatch);
  }, []);

  const onChange = (time, timeString) => {
    setTime(dayjs(timeString, 'HH:mm:ss'));
  };

  const handlePost = () => {
    setModal({ ...modalStaus, status: false, type: '' });
    let errorArray = [];
    if (!name) {
      errorArray.push('Please enter name to continue');
    }
    if (!venue) {
      errorArray.push('Please enter venue to continue');
    }
    if (!timeVal) {
      errorArray.push('Please enter time to continue');
    }
    if (!des) {
      errorArray.push('Please enter description to continue');
    }
    if (!date) {
      errorArray.push('Please enter date to continue');
    }
    if (errorArray.length > 0) {
      Swal.fire({
        title: `Please Fill all the required fields to continue`,
        // text: "You won't be able to revert this!",
        icon: 'warning',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Yes',
      });
    } else {
      if (modalStaus.type === 'create') {
        createAD(
          {
            name: name,
            time: timeVal,
            venue: venue,
            date: date,
            description: des,
          },
          dispatch
        );
      } else {
        updateAd(
          id,
          {
            name: name,
            time: timeVal,
            venue: venue,
            date: date,
            description: des,
          },
          dispatch
        );
      }
    }
  };

  useEffect(() => {
    console.log(adData, 'adDA');
    if (getAdStatus === true) {
      setAdData(adData);
    } else if (getAdStatus === false) {
      setAdData([]);
    }
  }, [getAdStatus]);

  useEffect(() => {
    if (updateAdStatus === true) {
      Swal.fire({
        title: `Advertisement has been successfully updated`,
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    } else if (updateAdStatus === false) {
      Swal.fire({
        title: `Error while updating Advertisement `,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    }
  }, [updateAdStatus]);

  useEffect(() => {
    if (deleteAdStatus === true) {
      Swal.fire({
        title: `Advertisement has been successfully deleted`,
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    } else if (deleteAdStatus === false) {
      Swal.fire({
        title: `Error while deleting Advertisement `,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    }
  }, [deleteAdStatus]);

  useEffect(() => {
    if (adCreateStatus === true) {
      Swal.fire({
        title: `Advertisement has been successfully Created`,
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    } else if (adCreateStatus === false) {
      Swal.fire({
        title: `Error while creating an Advertisement `,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      }).then((result) => {
        if (result.isConfirmed) {
          initailizeAd(dispatch);
        } else {
          initailizeAd(dispatch);
        }
      });
    }
  }, [adCreateStatus]);

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      //   width: '40%',

      key: 'name',
      render: (text) => <a>{text}</a>,
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
      render: (datas) => (
        <>
          <span>{moment(datas).format('YYYY-MM-DD')}</span>
        </>
      ),
    },
    {
      title: 'Time',
      dataIndex: 'time',
      key: 'time',
    },

    {
      title: 'Venue',
      key: 'venue',
      dataIndex: 'venue',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: 'Created Date',
      dataIndex: 'createdAt',
      key: 'createdAt',
    },
    {
      title: 'Action',
      //   width: '70%',
      key: 'action',
      render: (datas, record) => (
        <Space size="middle">
          <Button
            style={{ backgroundColor: 'orange', color: 'white' }}
            onClick={(e, record) => {
              console.log(
                record,
                datas.date,
                moment(datas.date).locale('en-Us').format('YYYY-MM-DD')
              );
              setModal({ ...modalStaus, status: true, type: 'edit' });
              setName(datas.name);
              setDate(moment(datas.date));
              setTime(datas.time);
              setVenue(datas.venue);
              setDesc(datas.description);
              setId(datas.id);
            }}
          >
            Edit
          </Button>
          <Button
            style={{ backgroundColor: 'red', color: 'white' }}
            onClick={(e, record) => {
              Swal.fire({
                title: `Are You Sure`,
                // text: "You won't be able to revert this!",
                icon: 'warning',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Yes',
              }).then((result) => {
                if (result.isConfirmed) {
                  deleteAd(datas.id, dispatch);
                }
              });
            }}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];
  return (
    <Row gutter={[24, 0]}>
      <Col xs={24} className="mb-24">
        <Card
          className="header-solid h-full ant-card-p-0"
          title={
            <>
              <Row
                gutter={[24, 0]}
                className="ant-row-flex ant-row-flex-middle"
              >
                <Col xs={24} md={12}>
                  <h6 className="font-semibold m-0">Advertisements</h6>
                </Col>
                <Col xs={24} md={12} className="d-flex">
                  <Button
                    type="primary"
                    onClick={() => {
                      setDate(null);
                      setDesc('');
                      setName('');
                      setVenue('');
                      setTime('00:00');
                      setModal({ ...modalStaus, status: true, type: 'create' });
                    }}
                  >
                    ADD NEW ADVERTISEMENT
                  </Button>
                </Col>
              </Row>
            </>
          }
        >
          {/* <Row gutter={[24, 0]}> */}
          {/* <Col span={24} md={12}> */}
          <Table
            columns={columns}
            dataSource={adDataVal}
            pagination={false}
            className="ant-border-space"
          />

          <Modal
            closable={false}
            title={
              modalStaus?.type === 'create'
                ? 'Let’s Post Advertisement'
                : 'Update Adverstisement'
            }
            centered
            visible={modalStaus.status}
            footer={null}
            // onOk={() => setOpen(false)}
            // onCancel={() => setOpen(false)}
            width={850}
          >
            <div
              className="custom-modal-content"
              style={{ backgroundColor: '#E6E7FF', height: '400px' }}
            >
              <div style={{ height: '30px' }}></div>
              <table style={{ width: '100%' }}>
                <tbody>
                  <tr>
                    <td>
                      <div style={{ width: '10px' }}></div>
                    </td>
                    <td style={{ textAlign: 'left' }}>
                      <span
                        style={{
                          color: '#070C83',
                          fontWeight: 600,
                          fontSize: '20px',
                        }}
                      >
                        Name
                      </span>
                    </td>

                    <td style={{ textAlign: 'left' }}>
                      <Input
                        value={name}
                        style={{ width: '650px' }}
                        onChange={(e) => setName(e.target.value)}
                      />
                    </td>
                    <td>
                      <div style={{ width: '10px' }}></div>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div style={{ height: '20px' }}></div>
              <Row>
                <table>
                  <tbody>
                    <tr>
                      <td>
                        <div style={{ width: '10px' }}></div>
                      </td>
                      <td style={{ textAlign: 'left' }}>
                        <span
                          style={{
                            color: '#070C83',
                            fontWeight: 600,
                            fontSize: '20px',
                          }}
                        >
                          Date
                        </span>
                      </td>

                      <td style={{ textAlign: 'right' }}>
                        &ensp;&ensp;{' '}
                        <DatePicker
                          value={date}
                          onChange={(date, dateString) =>
                            setDate(moment(dateString))
                          }
                        />
                      </td>
                      <td style={{ textAlign: 'left' }}>
                        &ensp;&ensp;&ensp;&ensp;
                        <span
                          style={{
                            color: '#070C83',
                            fontWeight: 600,
                            fontSize: '20px',
                          }}
                        >
                          Time
                        </span>
                      </td>
                      {/* <td>
                  <div style={{ width: '10px' }}></div>
                </td> */}
                      <td style={{ textAlign: 'right' }}>
                        &ensp;&ensp;
                        <TimePicker
                          allowClear={false}
                          format={format}
                          value={moment(timeVal, format)}
                          onChange={(value, dateString) => {
                            setTime(dateString);
                          }}
                        />
                      </td>
                      <td style={{ textAlign: 'left' }}>
                        &ensp;&ensp;
                        <span
                          style={{
                            color: '#070C83',
                            fontWeight: 600,
                            fontSize: '20px',
                          }}
                        >
                          Venue
                        </span>
                      </td>
                      <td>
                        <div style={{ width: '10px' }}></div>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <Input
                          value={venue}
                          style={{ width: '100%%' }}
                          onChange={(e) => setVenue(e.target.value)}
                        />
                      </td>
                    </tr>
                  </tbody>
                </table>
              </Row>

              <div style={{ height: '20px' }}></div>
              <table>
                <tbody>
                  <tr>
                    <td>
                      <div style={{ width: '10px' }}></div>
                    </td>
                    <td style={{ textAlign: 'left' }}>
                      <span
                        style={{
                          color: '#070C83',
                          fontWeight: 600,
                          fontSize: '20px',
                        }}
                      >
                        Description
                      </span>
                    </td>

                    <td style={{ textAlign: 'left', width: '600px' }}>
                      <TextArea
                        value={des}
                        style={{ height: '150px' }}
                        onChange={(e) => setDesc(e.target.value)}
                      />
                    </td>
                  </tr>
                </tbody>
              </table>
              <div style={{ marginLeft: '200px', marginTop: '30px' }}>
                <Space size={150}>
                  <Button
                    onClick={handlePost}
                    style={{
                      background: 'linear-gradient(to right, #000AF3, #69D0F0)', // Replace with your desired gradient colors
                      border: 'none',
                      color: 'white',
                      width: '150px',
                    }}
                  >
                    {modalStaus.type === 'create' ? 'Post' : 'Update'}
                  </Button>
                  <Button
                    onClick={() =>
                      setModal({ ...modalStaus, status: false, type: '' })
                    }
                    style={{
                      background: 'linear-gradient(to right, #000AF3, #69D0F0)', // Replace with your desired gradient colors
                      border: 'none',
                      color: 'white',
                      width: '150px',
                    }}
                  >
                    Cancel
                  </Button>
                </Space>
              </div>
            </div>
          </Modal>
        </Card>
      </Col>
    </Row>
  );
};

export default Advertisement;
