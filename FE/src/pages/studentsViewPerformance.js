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
  Rate,
} from 'antd';
import React, { useEffect, useState } from 'react';
import {
  createAD,
  deleteAd,
  getAllAds,
  getAllStudents,
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
import Search from 'antd/lib/input/Search';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
dayjs.extend(customParseFormat);
const format = 'HH:mm';
const limit = 10;
const desc = ['terrible', 'bad', 'normal', 'good', 'wonderful'];

// import DatePicker from 'react-datepicker';

const StudentViewPerformance = () => {
  const getStudentStatus = useSelector(
    (state) => state?.studentReducer?.getStudentStatus
  );
  const [value, setValue] = useState(3);

  const [search, setSearch] = useState('');
  const getStudentData = useSelector(
    (state) => state?.studentReducer?.getStudentData
  );

  React.useEffect(() => {
    const getData = setTimeout(() => {
      getAllStudents({ page: 1, limit, search }, dispatch);
    }, 500);

    return () => clearTimeout(getData);
  }, [search]);

  const onSearch = (e) => {
    setSearch(e.target.value);
  };

  const dispatch = useDispatch();

  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);

  const [studentData, setData] = useState([]);

  useEffect(() => {
    if (getStudentStatus) {
      setData(getStudentData?.results?.data || []);
      setTotalCount(getStudentData?.results?.count || 0);
    } else {
      setData([]);
      setTotalCount(0);
    }
  }, [getStudentStatus]);

  const columns = [
    {
      title: 'Index No',
      dataIndex: 'indexNo',
      key: 'indexNo',
    },
    {
      title: 'Student',
      //   dataIndex: 'lastName',
      key: 'lastName',
      render: (datas, record) => (
        <>
          <span>{`${datas?.firstName} ${datas.lastName}`}</span>
        </>
      ),
    },

    {
      title: 'Badge',
      key: 'ranks',
      render: (datas, record) => (
        <>
          <span>
            {datas?.ranks === 1
              ? 'Gold 🥇'
              : datas?.ranks === 2
              ? 'Silver 🥈'
              : datas?.ranks === 3
              ? 'Bronze 🥉'
              : ''}
          </span>
        </>
      ),
    },

    {
      title: 'Rank',
      key: 'ranks',
      render: (datas, record) => (
        <>
          <span>{datas?.ranks || ''}</span>
        </>
      ),
    },
    {
      title: 'Points',
      dataIndex: 'total_score',
      key: 'total_score',
    },
    {
      title: 'Type',
      key: 'Type',
      render: (datas, record) => (
        <>
          {datas?.performance_level || ''}
        </>
      ),
    },
    {
      title: 'Feedback',
      key: 'Type',
       render: (datas, record) => (
        <>
      <a href="http://localhost:3000/api/fetchData" class="button">View Feedback</a>
       </>
      ),
    },
  ];
  useEffect(() => {
    getAllStudents({ page: 1, limit, search }, dispatch);
  }, []);

  const exportPDF = () => {
    const unit = 'pt';
    const size = 'A4'; // Use A1, A2, A3 or A4
    const orientation = 'portrait'; // portrait or landscape

    const marginLeft = 40;
    const doc = new jsPDF(orientation, unit, size);

    doc.setFontSize(15);

    const title = 'Students Details Report';
    const headers = [['Index No', 'Name', 'Points', 'Rank']];

    const data = studentData.map((elt) => [
      elt.indexNo,
      `${elt.firstName} ${elt.lastName}`,
      elt.total_score,
      elt?.ranks || '',
    ]);

    let content = {
      startY: 50,
      head: headers,
      body: data,
    };

    doc.text(title, marginLeft, 40);
    doc.autoTable(content);
    doc.save(
      `students_report_for_date ${moment(new Date()).format('YYYY-MM-DD')}.pdf`
    );
  };
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
                  <h6 className="font-semibold m-0">Students Performance</h6>
                </Col>
              </Row>
            </>
          }
        >
          <Row gutter={[24, 0]} className="ant-row-flex ant-row-flex-middle">
            <Col xs={24} md={12}></Col>
            <Col xs={24} md={12} className="d-flex">
              <Button onClick={exportPDF}>EXPORT TO PDF</Button>
            </Col>
          </Row>
          <Row>
            <Input
              placeholder="search by index no"
              onChange={onSearch}
              style={{ width: 400 }}
            />
          </Row>
          {/* <Row gutter={[24, 0]}> */}
          {/* <Col span={24} md={12}> */}
          <Table
            pagination={{ pageSize: limit, total: totalCount }}
            columns={columns}
            dataSource={studentData}
            // pagination={true}
            onChange={(page, pageSize) => {
              setPage(page.current);
              getAllStudents({ page: page.current, limit, search }, dispatch);
            }}
            className="ant-border-space"
          />
        </Card>
      </Col>
    </Row>
  );
};

export default StudentViewPerformance;
