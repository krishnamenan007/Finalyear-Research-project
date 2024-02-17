import {Card, Col, Row, Tabs} from 'antd';
import {useEffect, useState} from 'react';
import Gpa from './studentsAddPerformance/gpa';
import SelfLearning from './studentsAddPerformance/selfLearning';
import Club from './studentsAddPerformance/club';
import Event from './studentsAddPerformance/events';
import Sport from './studentsAddPerformance/sport';

const StudentPerformanceTabs = () => {
    const [tab, setTab] = useState('2');

    useEffect(() => {
        let url = window.location.href;
        const splittedURL = url.split('/')[4];

        console.log(splittedURL, 'uerll');
        if (splittedURL === 'gpa') {
            setTab('1');
        } else if (splittedURL === 'selfLearning') {
            setTab('2');
        } else if (splittedURL === 'club') {
            setTab('4');
        } else if (splittedURL === 'event') {
            setTab('5');
        } else if (splittedURL === 'sport') {
            setTab('3');
        }
    }, []);

    const handleTabs = (key) => {
        setTab(key);
    };
    return (
        <div style={{display: 'flex', justifyContent: 'center', width: '100%',height:'100%',alignItems:'start'}} className="addPerformanceBackImage">
            <div className="mb-24" style={{width: '50%',paddingTop:'50px'}}>
                <h1 className="title-style"
                    style={{fontSize: '30px', fontWeight: 'bold', textAlign: 'center', color: '#070C83'}}>
                    Come let’s analyze your Performance!
                </h1>
                <p className="title-style"
                   style={{fontSize: '16px', fontWeight: 'normal', textAlign: 'left', color: '#070C83',padding :'20px'}}>
                    It may take some time to achieve your goals, but eventually, you will. So be patient and work hard
                    for your dreams, and they will come to you.
                </p>
            </div>

            <div className="mb-24" style={{width: '50%',height:'80%'}}>
                <Card
                    style={{width: '100%',backgroundColor:'transparent',zIndex:'1',boxShadow:'none',border:'none'}}
                    className="header-solid h-full ant-card-p-0"
                    title={
                        <>
                            <div
                                style={{padding: '0px 24px', width: '100%'}}
                                className="ant-row-flex ant-row-flex-middle"
                            >
                                <div style={{width: '100%'}} className="ant-col ant-col-24">
                                    <Tabs onChange={handleTabs} activeKey={tab} size="large" style={{width: '100%'}}>
                                        <Tabs.TabPane tab="GPA" key="1" style={{width: '100%'}}>
                                            <Gpa/>
                                        </Tabs.TabPane>
                                        <Tabs.TabPane tab="Self Learning" key="2">
                                            <SelfLearning/>
                                        </Tabs.TabPane>
                                        <Tabs.TabPane tab="Sport" key={3}>
                                            <Sport/>
                                        </Tabs.TabPane>
                                        <Tabs.TabPane tab="Club" key={4}>
                                            <Club/>
                                        </Tabs.TabPane>
                                        <Tabs.TabPane tab="Events" key={5}>
                                            {' '}
                                            <Event/>
                                        </Tabs.TabPane>
                                    </Tabs>
                                </div>
                            </div>
                        </>
                    }
                ></Card>
            </div>
        </div>
    );
};

export default StudentPerformanceTabs;
