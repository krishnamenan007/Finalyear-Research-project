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
    ConfigProvider,
} from 'antd';
import React, {useEffect, useState} from 'react';
import {
    createAD,
    deleteAd,
    getAllAds,
    initailizeAd,
    updateAd,
} from '../redux/action';
import Swal from 'sweetalert2';
import moment from 'moment';
import {useDispatch, useSelector} from 'react-redux';
import TextArea from 'antd/lib/input/TextArea';
import 'react-time-picker/dist/TimePicker.css';
import 'react-clock/dist/Clock.css';
import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';
import {useHistory} from 'react-router-dom/cjs/react-router-dom';
import GpaSVG from './Icons/gpaSVG';
import GpaIcon from './Icons/gpaIcon';
import SelfIcon from './Icons/selfLearningIcon';
import SportIcon from './Icons/sportsIcon';
import ClubIcon from './Icons/clubIcon';
import EventIcon from './Icons/eventIcon';
import ArrowIcon from '../assets/images/arrow.png';
import Banner2Icon from './Icons/banner2Icon';

const StudentAddPerformance = () => {
    const history = useHistory();

    return (
        <Row gutter={[24, 0]} className="performanceBackImage" style={{height: '120%',width:"98%",marginLeft:"20px",borderRadius:"20px"}}>
            <Col xs={24} className="mb-24" style={{display: 'flex', justifyContent: 'center'}}>
                <Card
                    style={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'start',
                        width: '80%',
                        backgroundColor: 'transparent',
                        border: 'none',
                        boxShadow: 'none',
                        paddingTop: '5%'
                    }}
                    className="header-solid ant-card-p-0"
                    title={
                        <>
                            <Row
                                gutter={[24, 0]}
                                className="ant-row-flex ant-row-flex-middle"
                            >
                                <Col xs={24} md={12}></Col>
                            </Row>
                        </>
                    }
                >
                    <ConfigProvider
                        theme={{
                            token: {
                                fontFamily: 'Poppins',
                                backgroundColor: 'white',
                            },
                        }}
                    >
                        <div style={{
                            width: '100%',
                            fontFamily: 'Poppins !important',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center'
                        }}>
                            <Row>
                                <Col>
                  <span
                      style={{
                          color: '#021D42',
                          fontSize: '15px',
                          fontWeight: 600,
                      }}
                  >
                    Hey There,
                  </span>
                                </Col>
                            </Row>
                            <Row>
                                <Col>
                  <span
                      style={{
                          color: '#021D42',
                          fontSize: '25px',
                          fontWeight: 900,
                      }}
                  >
                    Come let’s analyze your Performance!
                  </span>
                                </Col>
                            </Row>
                            <Row>
                                <Col>
                                    <p
                                        style={{
                                            color: '#021D42',
                                            fontSize: '20px',
                                            fontWeight: 600,
                                        }}
                                    >
                                        <p align="justify">Always bear in mind that achievements are like the guiding stars along the journey of progress. They are tangible proof of your dedication and unwavering commitment to self-improvement. By relentlessly pursuing excellence and continually honing your skills, you not only advance in your current role but also open doors to exciting opportunities in your chosen field. Your accomplishments become a powerful portfolio, a testament to your resilience and determination. So, embrace the grind, stay resolute in your pursuit of growth, and you'll undoubtedly pave the way for a promising and deeply fulfilling future.</p>
                                    </p>
                                </Col>
                            </Row>

                            <Row>
                                <Col>
                  <span
                      style={{
                          color: '#021D42',
                          fontSize: '25px',
                          fontWeight: 600,
                      }}
                  >
                   <b>Join our community to enhance your skills! Take a step towards your GOALS.</b>
                  </span>
                                </Col>
                            </Row>
                            <Row
                                style={{
                                    width: '80%',
                                    marginLeft: '5%',
                                    marginTop: '50px',
                                    display: 'flex',

                                    flexWrap: 'wrap', // Allow items to wrap to the next line
                                    justifyContent: 'space-around', // Center items horizontally
                                }}
                            >
                                {/* Repeat this block for each icon */}
                                <div
                                    onClick={() => history.push('/addPerformance/gpa')}
                                    style={{
                                        backgroundColor: '#E6E7FF',
                                        position: 'relative',
                                        width: '100px',
                                        borderRadius: '10px',
                                        color: '#070C83',
                                        // Adjust the width for different screen sizes
                                        margin: '10px', // Add some margin between items
                                        display: 'flex',
                                        flexDirection: 'column', // Arrange icon and name in a column
                                        alignItems: 'center', // Center items horizontally
                                        cursor: 'pointer',
                                    }}
                                >
                                    <GpaIcon/>
                                    <button
                                        onClick={() => history.push('/addPerformance/gpa')}
                                        style={{
                                            backgroundColor: '#E6E7FF', // Button background color
                                            color: '#070C83', // Text color
                                            fontWeight: 'bold', // Bold text
                                            padding: '5px 10px', // Adjust padding as needed
                                            border: 'none', // Remove border
                                            borderRadius: '5px', // Add rounded corners
                                            cursor: 'pointer',
                                            width: '100%', // Make the button width 100% to be responsive
                                            marginTop: '3%',
                                        }}
                                    >
                                        GPA
                                    </button>


                                </div>

                                <div
                                    onClick={() => history.push('/addPerformance/selflearning')}
                                    style={{
                                        backgroundColor: '#E6E7FF',
                                        position: 'relative',
                                        width: '100px',
                                        borderRadius: '10px',
                                        // Adjust the width for different screen sizes
                                        color: '#070C83',
                                        margin: '10px', // Add some margin between items
                                        display: 'flex',
                                        flexDirection: 'column', // Arrange icon and name in a column
                                        alignItems: 'center', // Center items horizontally
                                        cursor: 'pointer',
                                        marginTop: '2px',
                                    }}
                                >
                                    <SelfIcon style={{position: 'absolute', top: '0', left: '0'}}/>
                                    <button
                                        onClick={() => history.push('/addPerformance/selflearning')}
                                        style={{
                                            backgroundColor: '#E6E7FF', // Button background color
                                            color: '#070C83', // Text color
                                            fontWeight: 'bold', // Bold text
                                            padding: '5px 10px', // Adjust padding as needed
                                            border: 'none', // Remove border
                                            borderRadius: '5px', // Add rounded corners
                                            cursor: 'pointer',
                                            width: '100%',
                                            marginTop: '2%', // Make the button width 100% to be responsive
                                        }}
                                    >
                                        Self-Learning
                                    </button>
                                </div>

                                <div
                                    onClick={() => history.push('/addPerformance/sport')}
                                    style={{
                                        backgroundColor: '#E6E7FF',
                                        position: 'relative',
                                        width: '100px',
                                        borderRadius: '10px',
                                        // Adjust the width for different screen sizes
                                        margin: '10px', // Add some margin between items
                                        display: 'flex',
                                        flexDirection: 'column', // Arrange icon and name in a column
                                        alignItems: 'center', // Center items horizontally
                                        cursor: 'pointer',
                                    }}

                                >
                                    <SportIcon style={{position: 'absolute', top: '0', left: '0'}}/>
                                    <button
                                        onClick={() => history.push('/addPerformance/sport')}
                                        style={{
                                            backgroundColor: '#E6E7FF', // Button background color
                                            color: '#070C83', // Text color
                                            fontWeight: 'bold', // Bold text
                                            padding: '5px 10px', // Adjust padding as needed
                                            border: 'none', // Remove border
                                            borderRadius: '5px', // Add rounded corners
                                            cursor: 'pointer',
                                            width: '50%',
                                            marginTop: '4%', // Make the button width 100% to be responsive
                                        }}

                                    >
                                        Sport
                                    </button>
                                </div>

                                <div
                                    onClick={() => history.push('/addPerformance/event')}
                                    style={{
                                        backgroundColor: '#E6E7FF',
                                        position: 'relative',
                                        width: '100px',
                                        borderRadius: '10px',
                                        // Adjust the width for different screen sizes
                                        margin: '10px', // Add some margin between items
                                        display: 'flex',
                                        flexDirection: 'column', // Arrange icon and name in a column
                                        alignItems: 'center', // Center items horizontally
                                        cursor: 'pointer',
                                    }}
                                >
                                    <EventIcon style={{position: 'absolute', top: '0', left: '0'}}/>
                                    <button
                                        onClick={() => history.push('/addPerformance/event')}
                                        style={{
                                            backgroundColor: '#E6E7FF', // Button background color
                                            color: '#070C83', // Text color
                                            fontWeight: 'bold', // Bold text
                                            padding: '5px 10px', // Adjust padding as needed
                                            border: 'none', // Remove border
                                            borderRadius: '5px', // Add rounded corners
                                            cursor: 'pointer',
                                            width: '50%',
                                            marginTop: '4%', // Make the button width 100% to be responsive
                                        }}

                                    >
                                        Event
                                    </button>


                                </div>
                                <div
                                    onClick={() => history.push('/addPerformance/club')}
                                    style={{
                                        backgroundColor: '#E6E7FF',
                                        position: 'relative',
                                        width: '100px',
                                        borderRadius: '10px',
                                        // Adjust the width for different screen sizes
                                        margin: '10px', // Add some margin between items
                                        display: 'flex',
                                        flexDirection: 'column', // Arrange icon and name in a column
                                        alignItems: 'center', // Center items horizontally
                                        cursor: 'pointer',
                                        marginBottom: '1%',
                                    }}
                                >

                                    <ClubIcon style={{position: 'absolute', top: '0', left: '0'}}/>
                                    <button
                                        onClick={() => history.push('/addPerformance/club')}
                                        style={{
                                            backgroundColor: '#E6E7FF', // Button background color
                                            color: '#070C83', // Text color
                                            fontWeight: 'bold', // Bold text
                                            padding: '5px 10px', // Adjust padding as needed
                                            border: 'none', // Remove border
                                            borderRadius: '5px', // Add rounded corners
                                            cursor: 'pointer',
                                            marginTop: '2%',
                                            width: '50%', // Make the button width 100% to be responsive
                                        }}

                                    >
                                        club
                                    </button>

                                </div>

                            </Row>
                        </div>
                    </ConfigProvider>
                    <div style={{position: 'absolute', bottom: '0', right: '0',display:'flex',justifyContent:'flex-end',alignItems:'flex-end'}}>
                        <div className="performanceBackImageQuote" style={{maxWidth: '80%',padding:'20px',borderRadius:'10px'}}>
                            <div  style={{color: 'black', fontSize: '15px', fontWeight: 'bold',backgroundColor:'white',borderRadius:'10px',padding:'20px',display:'flex',justifyContent:'center',alignItems:'center'}}>
                                <div>
                                    Always set realistic goals and things to do in your day. Make a list and force yourself
                                    to do it. This will help you with deadlines, and your studies will never pile up. Always
                                    say “I can do it” to yourself, and believe me, you will.

                                </div>
                                <div>
                                    <img src={ArrowIcon} style={{width:'20px',height:'20px',marginLeft:'10px'}}/>
                                </div>
                            </div>

                        </div>
                    </div>
                </Card>
            </Col>
        </Row>
    );
};

export default StudentAddPerformance;
