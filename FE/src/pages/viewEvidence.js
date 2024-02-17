import React, { useState, useEffect } from 'react';
import {
    Card,
    Col,
    Row,
    Table,
    Button
} from 'antd';
import axios from 'axios';

const ViewEvidence = () => {
    const [update, setUpdate] = useState(0);

    async function handleApprove(record) {
        console.log("Approve Button clicked for record: ", record);
        const token = localStorage.getItem('token');
        try {
            // Assuming 'record' contains 'id' as the item_id and 'type' to distinguish between selflearning and sports
            await axios.post(`http://localhost:4000/approvel/updateApproved`,
                {
                    item_id: record.item_id, // send item_id in the request body
                    type: record.type // send type in the request body
                },
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                        'content-type': 'application/json',
                    },
                }
            );
            // Refetch the data to reflect the changes
            fetchData();
        } catch (error) {
            console.error(error);
        }
    }

    async function handleCancel(record) {
        console.log("Cancel Button clicked for record: ", record);
        const token = localStorage.getItem("token");
        try {
            console.log("Sending request to updateRejected with record:", record);
            const response = await axios.post(
                `http://localhost:4000/approvel/updateRejected`,
                {
                    item_id: record.item_id,
                    type: record.type,
                },
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                        "content-type": "application/json",
                    },
                }
            );
            fetchData();

        } catch (error) {
            console.error("Error in handleCancel function:", error);
        }
    }


    const [data, setData] = useState([]);

    const fetchData = async () => {
        const token = localStorage.getItem('token');
        try {
            const response = await axios.get(`http://localhost:4000/approvel`, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'content-type': 'application/json',
                },
            });
            setData(response.data.results.data);
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const columns = [
        {
            title: 'Index No',
            dataIndex: 'index_no',
            key: 'index_no',
        },
        {
            title: 'Student',
            dataIndex: 'first_name',
            key: 'first_name',
        },
        {
            title: 'Level',
            dataIndex: 'level',
            key: 'level',
        },
        {
            title: 'Evidence Documents',
            dataIndex: 'file_data',
            key: 'file_data',
            render: (file_data) => {
                // Converting the file data to Blob and creating an object URL
                const blob = new Blob([new Uint8Array(file_data.data)], { type: 'application/pdf' });
                const url = URL.createObjectURL(blob);
                return <a href={url} download="file.pdf">Download</a>;
            },
        },
        {
            title: 'Actions',
            key: 'actions',
            render: (text, record) => (
                <>
                    <Button type="primary" onClick={() => handleApprove(record)} style={{ marginRight: '10px' }}>
                        Approve
                    </Button>
                    <Button type="default" onClick={() => handleCancel(record)}>
                        Cancel
                    </Button>
                </>
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
                                    <h6 className="font-semibold m-0">View Evidence</h6>
                                </Col>
                            </Row>
                        </>
                    }
                >
                    <Table columns={columns} dataSource={data} pagination={false} className="ant-border-space" />
                </Card>
            </Col>
        </Row>
    );
};

export default ViewEvidence;
