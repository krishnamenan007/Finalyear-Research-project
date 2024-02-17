import {Card, Col, Row, Tooltip} from 'antd';

import LineChart from '../components/chart/LineChart';
import advertisement from "../assets/images/Advertisment.png";
import Paragraph from "antd/lib/typography/Paragraph";
import DatePicker from "react-horizontal-datepicker";
import moment from "moment/moment";
import nextBtn from "../assets/images/Next.png";

const GraphComponent = () => {
    return (
        <div className="layout-content" style={{height: '100%',background: "#fff"}}>
            <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center',flexWrap:"wrap", flexDirection: 'row',height:'100%'}}>
                <Row style={{width: '60%',height:'100%'}} gutter={24}>
                    <Col xs={24} sm={24} md={24} lg={24} xl={24} className="mb-24">
                        <Card bordered={false} className="criclebox h-full" style={{display: 'flex', flexDirection: 'column', gap: "50px", height: '100%'}}>
                            <div style={{display: 'flex', flexDirection: 'column', gap: "10px",height:'40%'}}>
                                <h1 style={{color: "#070C83", fontWeight: 600, fontSize: "30px"}}>
                                    Let’s Post Advertisement</h1>
                                <div style={{color: "#070C83", fontWeight: 400, fontSize: "15px"}}>
                                    It may take some time to achieve your goals, but eventually, you will. So be patient and
                                    work hard for your dreams, and they will come to you.It may take some time to achieve
                                    your goals, but eventually,
                                </div>
                            </div>
                            <div style={{background: "#E6E7FF", borderRadius: "10px", padding: "20px"}}>
                                <LineChart/>
                            </div>

                        </Card>
                    </Col>
                </Row>
                <div style={{width: '30%'}}>
                    <div style={{
                        width: "80%",
                        margin: "auto",
                        textAlign: "center",
                        backgroundColor: "#E6E7FF",
                        borderRadius: "10px",
                        padding: "20px",
                        height: "100%"
                    }}>
                        <Row gutter={[24, 0]}>

                            <Col xs={24} sm={24} md={12} lg={12} xl={24} className="mb-24">

                                <div className=" h-full">
                                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                                        <img src={advertisement} style={{width: "70px", height: "70px"}}/>
                                        <div style={{
                                            display: 'flex',
                                            flexDirection: 'column',
                                            gap: "5px",
                                            marginLeft: "20px",
                                            marginTop: "20px"
                                        }}>
                                            <div
                                                style={{color: "#070C83", fontWeight: 600, fontSize: "30px"}}
                                            >
                                                Hey There
                                            </div>
                                            <div
                                                style={{color: "#070C83", fontWeight: 600, fontSize: "15px"}}
                                            >
                                                {/* <Title level={5}>Active Users</Title> */}
                                                <Paragraph
                                                    className="lastweek"
                                                    style={{color: "#070C83", fontWeight: 600, fontSize: "15px"}}
                                                >
                                                    You need to focus on this!
                                                </Paragraph>
                                            </div>
                                        </div>
                                    </div>

                                    <Row>
                                        <Col xs={24} sm={24} md={12} lg={6} xl={23} className=""
                                             style={{
                                                 background: "#fff",
                                                 borderRadius: "10px",
                                                 padding: "20px",
                                                 fontSize: "15px"
                                             }}>
                                            <div style={{display: 'flex', flexDirection: 'column', gap: "10px"}}>
                                          <span>
                                              You are on the right track. By starting to study for the exam earlier, you may be able to retain more knowledge on exam day.
                                          </span>
                                            </div>
                                            <div style={{
                                                display: 'flex',
                                                justifyContent: 'flex-end',
                                                marginTop: "20px"
                                            }}>
                                                You are on the right track. By starting to study for the exam earlier,
                                                you may be able to retain more knowledge on exam day.
                                            </div>
                                        </Col>
                                    </Row>
                                </div>
                            </Col>
                        </Row>


                    </div>
                </div>
            </div>
        </div>
    );
};

export default GraphComponent;
