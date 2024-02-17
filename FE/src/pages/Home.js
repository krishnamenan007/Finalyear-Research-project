import {useEffect, useState} from "react";

import {
    Card,
    Col,
    Row,
    Typography,
    /* Tooltip,
     Progress,
     Upload,
     message,
     Button,
     Timeline,
     Radio,*/
} from "antd";
// import {
//   ToTopOutlined,
//   MenuUnfoldOutlined,
//   RightOutlined,
// } from "@ant-design/icons";
import Paragraph from "antd/lib/typography/Paragraph";
import {Tooltip} from 'antd';
import nextBtn from "../assets/images/Next.png";
//  import Echart from "../components/chart/EChart";

import DatePicker from "react-horizontal-datepicker";
import moment from "moment";
import {useDispatch, useSelector} from "react-redux";
import {getAdsByDate} from "../redux/action";
import moments from "moment-timezone";
import ReactHorizontalDatePicker from "react-horizontal-strip-datepicker";
import "react-horizontal-strip-datepicker/dist/ReactHorizontalDatePicker.css";
import advertisement from "../assets/images/Advertisment.png";

const targetTimezone = "Asia/Kolkata"; // Replace with your desired timezone

function Home() {
    const {Title, Text} = Typography;
    const dispatch = useDispatch();
    const [selectedDay, setSelectedDay] = useState();
    const adByDateStatus = useSelector(
        (state) => state?.advertisementReducer?.adByDateStatus
    );

    const adByDateData = useSelector(
        (state) => state?.advertisementReducer?.adByDateData
    );

    const [data, setData] = useState([]);
    const onChange = (e) => console.log(`radio checked:${e.target.value}`);

    const [reverse, setReverse] = useState(false);


    useEffect(() => {
        if (adByDateStatus) {
            console.log(adByDateData, "adByDateData>>");
            setData(adByDateData);
        } else if (adByDateStatus === false) {
            setData([]);
        }
    }, [adByDateStatus]);

    const getSelectDayStr = (val) => {
        console.log(val, "sele2");
        setSelectedDay(val);

        const dateObject = new Date(val);

        const formattedDate = dateObject.toLocaleString("en-US");
        if (formattedDate !== "Invalid Date") {
            getAdsByDate(formattedDate, dispatch);
        } else {
            getAdsByDate("", dispatch);
        }
    };

    return (
        <>
            <div className="layout-content "
                 style={{padding: "0px", height: "100%", backgroundRepeat: "no-repeat", backgroundSize: "cover"}}>
                {/*<Row className="rowgap-vbox" gutter={[24, 0]}>*/}
                {/* */}
                {/*</Row>*/}

                <div
                    style={{
                        padding: "0px 24px 24px 24px",
                        width: "100%",
                        display: "flex",
                        flexWrap: "wrap",
                        justifyContent: "center",
                        height: "100%",
                        gap: "20px"
                    }}>
                    <div
                        className="homeBackImage"
                        style={{
                            width: "60%",
                            margin: "auto",
                            textAlign: "center",
                            height: "100%",
                            padding: "20px",
                            backgroundColor: "#BCBFFF",
                            borderRadius: "10px"
                        }}>
                        <div style={{width: "80%", margin: "auto", textAlign: "left", fontSize: "30px"}}>
                            <h2 style={{color: "#070C83", fontWeight: 900, fontSize: "30px",}}>
                                Come Join with us!
                            </h2>
                        </div>

                        <div style={{
                            width: "80%",
                            display: "flex",
                            flexDirection: "column",
                            gap: "20px",
                            margin: "auto",
                            textAlign: "left"
                        }}>
              <span>
                It may take some time to achieve your goals, but eventually, you will. So be patient and work hard for your dreams, and they will come to you.
              </span>
                            <span><p align="justify">Leveling up your skills today ensures a brighter career tomorrow. In the ever-evolving landscape of the professional world, the importance of continuous growth and development cannot be overstated. As industries adapt to technological advancements and market shifts, individuals who invest in honing their abilities will find themselves better equipped to thrive. Whether you're looking to advance in your current role or explore new career opportunities, the relentless pursuit of improvement is the key to success. Keep grinding, stay committed to self-improvement, and you'll pave the way for a promising and fulfilling future in your chosen field.</p>
                
              </span>
                       
                            <span>
                            <b>"Collect achievements, not just tasks. Each accomplishment is a step towards your dream career."</b>
              </span>
                        </div>

                        

                    


                    </div>

                    <div
                        id="home"
                        style={{
                        width: "30%",
                        margin: "auto",
                        textAlign: "center",
                        flexWrap: "wrap",
                        backgroundColor: "#BCBFFF",
                        borderRadius: "10px",
                        padding: "20px",
                        height: "100%"
                    }}>
                                <div className=" ">
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
                                                    You have somthing new!
                                                </Paragraph>
                                            </div>
                                        </div>
                                    </div>

                                        <div  className=""
                                             style={{background: "#fff", borderRadius: "10px", padding: "20px"}}>
                                            <DatePicker
                                                style={{background: "#070C83", color: "#fff"}}
                                                getSelectedDay={getSelectDayStr}
                                                endDate={100}
                                                selectDate={selectedDay}
                                                labelFormat={"MMMM"}
                                                color={"#374e8c"}
                                            />
                                        </div>
                                </div>

                                <div className="project-ant">
                                    <div>

                                        {selectedDay === "Invalid Date" ? (
                                            <></>
                                        ) : (
                                            <h2 style={{color: "#070C83", fontWeight: 600, fontSize: "20px"}}>
                                                {/*12    of Wednesday 2023  Format*/}
                                                {moment(selectedDay).format("dddd, MMMM Do YYYY")}
                                            </h2>
                                        )}
                                    </div>
                                    {/* <div className="ant-filtertabs">
                  <div className="antd-pro-pages-dashboard-analysis-style-salesExtra">
                    <Radio.Group onChange={onChange} defaultValue="a">
                      <Radio.Button value="a">ALL</Radio.Button>
                      <Radio.Button value="b">ONLINE</Radio.Button>
                      <Radio.Button value="c">STORES</Radio.Button>
                    </Radio.Group>
                  </div>
                </div> */}
                                </div>

                                {data.length > 0 ? (
                                    <div className="ant-list-box table-responsive" style={{width : "100%"}}>
                                        {data.map((d, index) => (
                                            <div key={index}
                                                 style={{
                                                     marginBottom: "10px",
                                                     display: "flex",
                                                     justifyContent: "space-between",
                                                     flexDirection: "row",
                                                     backgroundColor: "#FCFFF1",
                                                     padding: "10px",
                                                     borderRadius: "10px",
                                                     boxShadow: "0px 0px 10px 0px #ccc",
                                                     textAlign: "left",
                                                     width : "100%"
                                                 }}>
                                                <div style={{display: "flex", flexDirection: "column",width:"80%"}}>
                                                    <div style={{fontSize:'20px',fontWeight:'600'}} className="percent-progress">{d.name}</div>
                                                    <div className="percent-progress">{d.venue}</div>
                                                    <div className="percent-progress">
                                                        {d.description.toString().length > 50 ? (
                                                            <Tooltip title={d.description}>
                                                                {d.description.substring(0, 50)}...
                                                            </Tooltip>
                                                        ) : (
                                                            d.description
                                                        )}
                                                    </div>
                                                </div>
                                                <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                                                    <img src={nextBtn} width={'32px'} height={'32px'}/>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <>
                                        {" "}
                                        &ensp;&ensp;&ensp;{" "}
                                        <div style={{
                                            color: "#070C83",
                                            fontWeight: 600,
                                            fontSize: "20px",
                                            backgroundColor: "#fff",
                                            padding: "10px",
                                            borderRadius: "10px"
                                        }}>
                                            {selectedDay === "Invalid Date"
                                                ? "Oops seems like there is no events "
                                                : "Oops seems like there is no events on this day"}
                                        </div>
                                    </>
                                )}
                    </div>
                </div>
            </div>
        </>
    );
}

export default Home;
