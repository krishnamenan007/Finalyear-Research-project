import {useEffect, useState} from "react";
import React from 'react'; // Import React
import {
    Row,
    Col,
    Card,
    Button,
    List,
    Descriptions,
    Avatar,
    Radio,
    Space,
    Progress,
    Upload,
    message,
    Modal,
    Select,
    Input,
} from "antd";

import {
    FacebookOutlined,
    TwitterOutlined,
    InstagramOutlined,
    VerticalAlignTopOutlined,
} from "@ant-design/icons";

import BgProfile from "../assets/images/bg-profile.jpg";
import profilavatar from "../assets/images/face-1.jpg";
import convesionImg from "../assets/images/face-3.jpg";
import convesionImg2 from "../assets/images/face-4.jpg";
import convesionImg3 from "../assets/images/face-5.jpeg";
import convesionImg4 from "../assets/images/face-6.jpeg";
import convesionImg5 from "../assets/images/face-2.jpg";
import project1 from "../assets/images/home-decor-1.jpeg";
import project2 from "../assets/images/home-decor-2.jpeg";
import project3 from "../assets/images/home-decor-3.jpeg";
import {
    deleteStu,
    getProfileDetails,
    makeDeafultProfiles,
    updateProfileDetails,
    updateProfilePicDetails,
    updateStu,
} from "../redux/action";
import {useDispatch, useSelector} from "react-redux";
import GpaEdit from "./studentPerformanceEdit/gpa";
import Dragger from "antd/lib/upload/Dragger";
import {InboxOutlined, CloseOutlined} from "@ant-design/icons";
import Swal from "sweetalert2";
import UpdateProfile from "./updateProfile";
import {RFC_2822} from "moment";
import Column from "antd/lib/table/Column";

function Profile() {
    const dispatch = useDispatch();

    const [clubData, setClubData] = useState([]);
    const [selfLearningData, setSelfLearningData] = useState([]);
    const [sportData, setSportData] = useState([]);
    const [selectedClub, setSelectedClub] = useState([]);
    const [clubModal, setClubModal] = useState(false);
    const [eventModal, setEventModal] = useState(false);
    const [selfModal, setSelfModal] = useState(false);
    const [showModal, setShowModal] = useState(false);

    const handleClick = () => {
        setShowModal(!showModal); // Toggle showModal state
    };

    const closeModal = () => {
        setShowModal(false);
    };
    const [sportModal, setSportModal] = useState(false);
    const [selectedEventData, setSelectedEvent] = useState([]);
    const [selectedSportData, setSelectedSport] = useState([]);
    const [SelectedSelfLearning, serSelectedSelfLearning] = useState([]);

    const [eventData, setEventData] = useState([]);

    const [profileImgUrl, setProfileImgUrl] = useState(null);
    const [profileDataValue, setProfileData] = useState({});
    const [updateProfilePicStatusModal, setUpdateProfilePicStatus] =
        useState(false);

    const [openUpdateProfileInfoStatus, setUpdateInfoStatus] = useState(false);

    const [fileVal, setFile] = useState([]);
    const [picErrorStatus, setErrorStaus] = useState(false);

    const [imageURL, setImageURL] = useState(false);
    const [, setLoading] = useState(false);

    const profileData = useSelector(
        (state) => state?.studentReducer?.profileData
    );
    const profileStatus = useSelector(
        (state) => state?.studentReducer?.profileStatus
    );

    const updateProfileInfoStatus = useSelector(
        (state) => state?.studentReducer?.updateProfileInfoStatus
    );
    const [gpaModalStatus, setGpaModal] = useState(false);

    const deleteStuStatus = useSelector(
        (state) => state?.studentReducer?.deleteStuStatus
    );

    const updateStuStatus = useSelector(
        (state) => state?.studentReducer?.updateStuStatus
    );

    const updateProfilePicStatus = useSelector(
        (state) => state?.studentReducer?.updateProfilePicStatus
    );

    const getBase64 = (img, callback) => {
        const reader = new FileReader();
        reader.addEventListener("load", () => callback(reader.result));
        reader.readAsDataURL(img);
    };

    const beforeUpload = (file) => {
        const isJpgOrPng = file.type === "image/jpeg" || file.type === "image/png";
        if (!isJpgOrPng) {
            message.error("You can only upload JPG/PNG file!");
        }
        const isLt2M = file.size / 1024 / 1024 < 2;
        if (!isLt2M) {
            message.error("Image must smaller than 2MB!");
        }
        return isJpgOrPng && isLt2M;
    };

    const handleChange = (info) => {
        if (info.file.status === "uploading") {
            setLoading(false);
            return;
        }
        if (info.file.status === "done") {
            getBase64(info.file.originFileObj, (imageUrl) => {
                setLoading(false);
                setImageURL(false);
            });
        }
    };

    const handleFile = ({fileList}) => {
        console.log(fileList);
        setFile([...fileList]);
    };

    const dummyRequest = ({file, onSuccess}) => {
        setTimeout(() => {
            onSuccess("ok");
        }, 0);
    };

    const hideModal = () => {
        setUpdateInfoStatus(false);
    };

    useEffect(() => {
        if (updateStuStatus) {
            Swal.fire({
                title: `updated successfully`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
        } else if (updateStuStatus === false) {
            Swal.fire({
                title: `error when updating`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
        }

        makeDeafultProfiles(dispatch);
    }, [updateStuStatus]);

    useEffect(() => {
        if (deleteStuStatus) {
            Swal.fire({
                title: `deleted successfully`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
        } else if (deleteStuStatus === false) {
            Swal.fire({
                title: `error when deleting`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
        }

        makeDeafultProfiles(dispatch);
    }, [deleteStuStatus]);
    useEffect(() => {
        if (updateProfilePicStatus) {
            Swal.fire({
                title: `Profile picture has been updated successfully`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
            makeDeafultProfiles(dispatch);
        } else if (updateProfilePicStatus === false) {
            Swal.fire({
                title: `Error while updating profile picture`,
                // text: "You won't be able to revert this!",
                icon: "error",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
            makeDeafultProfiles(dispatch);
        }
    }, [updateProfilePicStatus]);

    useEffect(() => {
        if (updateProfileInfoStatus) {
            Swal.fire({
                title: `Profile Information has been updated successfully`,
                // text: "You won't be able to revert this!",
                icon: "success",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
            makeDeafultProfiles(dispatch);
        } else if (updateProfileInfoStatus === false) {
            Swal.fire({
                title: `Error while updating profile Information`,
                // text: "You won't be able to revert this!",
                icon: "error",
                showCancelButton: false,
                confirmButtonColor: "#3085d6",
                cancelButtonColor: "#d33",
                confirmButtonText: "OK",
            });
            makeDeafultProfiles(dispatch);
        }
    }, [updateProfileInfoStatus]);

    const pencil = [
        <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            key={0}
        >
            <path
                d="M13.5858 3.58579C14.3668 2.80474 15.6332 2.80474 16.4142 3.58579C17.1953 4.36683 17.1953 5.63316 16.4142 6.41421L15.6213 7.20711L12.7929 4.37868L13.5858 3.58579Z"
                className="fill-gray-7"
            ></path>
            <path
                d="M11.3787 5.79289L3 14.1716V17H5.82842L14.2071 8.62132L11.3787 5.79289Z"
                className="fill-gray-7"
            ></path>
        </svg>,
    ];


    useEffect(() => {
        makeDeafultProfiles(dispatch);
        getProfileDetails(dispatch);
    }, []);

    useEffect(() => {
        if (profileStatus === true) {
            if (profileData?.image) {
                console.log(profileData?.image?.data, "kkk");
                const buffer = Buffer.from(profileData?.image?.data);
                console.log(buffer, "kkk");
                const blob = new Blob([buffer], {
                    type: "image/jpeg",
                });

                const imageUrl = URL.createObjectURL(blob);
                setProfileImgUrl(imageUrl);
            } else {
                setProfileImgUrl(null);
            }
            setProfileData(profileData);

            let clubArray = profileData?.clubs
                ? profileData.clubs.map((club) => ({...club}))
                : [];
            let selfLearning = profileData?.selflearnings
                ? profileData.selflearnings.map((club) => ({...club}))
                : [];
            let events = profileData?.events
                ? profileData.events.map((club) => ({...club}))
                : [];
            let sports = profileData?.sports
                ? profileData?.sports.map((club) => ({...club}))
                : [];

            setClubData(clubArray);
            setSelfLearningData(selfLearning);
            setEventData(events);
            setSportData(sports);
        } else if (profileStatus === false) {
            setProfileData({});
            setProfileImgUrl(null);
        }
    }, [profileStatus]);

    const isMobile = window.innerWidth <= 768;

    return (
        <>
            {/*<div*/}
            {/*  className="profile-nav-bg"*/}
            {/*  style={{ backgroundImage: "url(" + BgProfile + ")" }}*/}
            {/*></div>*/}
            <div style={{padding: "10px", display: "flex", justifyContent: "center", gap: "1rem"}}
                 className={"profileBackImage"}>
                <div style={{background: "#fff", width: "40%", height: "50%", borderRadius: "30px"}}>

                    <Card
                        style={{background: "transparent"}}
                        className="card-profile-head"
                        bodyStyle={{display: "none"}}
                        title={
                            <Row justify="space-between" align="middle" gutter={[24, 0]}
                                 style={{display: "flex", width: "100%"}}>
                                <Col className="col-info" style={{
                                    width: "100%",
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center"
                                }}>
                                    <Avatar.Group style={{width: "80%"}}>
                                        <Avatar
                                            style={{width: "30%", height: "50%"}}
                                            size={64}
                                            shape="circle"
                                            src={profileImgUrl ? profileImgUrl : profilavatar}
                                        />

                                        <div className="avatar-info" style={{
                                            display: "flex",
                                            flexDirection: "column",
                                            gap: "10px",
                                            width: "100%",
                                            justifyContent: "center"
                                        }}>
                                            <h4 className="font-semibold m-0">
                                                {profileDataValue?.first_name} {profileDataValue?.last_name}
                                            </h4>
                                            <p>{profileDataValue?.role?.role_name || ""}</p>
                                            {profileDataValue.rank &&
                                            profileDataValue?.role?.role_code === "STUDENT" &&
                                            [1, 2, 3].includes(profileDataValue.rank) ? (
                                               <p> Rank:{profileDataValue?.rank || ""}</p>
                                                
                                            ) : (
                                                <></>
                                            )}

                                            {profileDataValue.rank &&
                                            profileDataValue?.role?.role_code === "STUDENT" &&
                                            [1, 2, 3].includes(profileDataValue.total_score) ? (
                                                <p>Rank:{profileDataValue?.rank || ""}</p>
                                            ) : (
                                                <></>
                                            )}

                                            {profileDataValue.rank &&
                                            profileDataValue?.role?.role_code === "STUDENT" &&
                                            [1, 2, 3].includes(profileDataValue.rank) ? (
                                                <p>
                                                    Badge:
                                                    {profileDataValue?.rank === 1
                                                        ? "Gold 🥇"
                                                        : profileDataValue?.rank === 2
                                                            ? "Silver 🥈"
                                                            : profileDataValue?.rank === 3
                                                                ? "Bronze 🥉"
                                                                : ""}
                                                </p>
                                            ) : (
                                                <></>
                                            )}

                                            <Button onClick={() => setUpdateProfilePicStatus(true)}>
                                                Update Profile Picture
                                            </Button>
                                        </div>
                                    </Avatar.Group>
                                </Col>

                            </Row>
                        }
                    ></Card>


                    <Row gutter={[24, 0]}>

                        <Col span={24} md={24} xl={24} className="mb-24">
                            <Card
                                style={{
                                    display: "flex",
                                    justifyContent: "center",
                                    flexDirection: "column",
                                    alignItems: "center"
                                }}
                                bordered={false} className="card-project"
                                title={<h6 className="font-semibold m-0">Profile Information</h6>}
                                // className="header-solid h-full card-profile-information"
                                extra={
                                    <Button type="link" onClick={() => setUpdateInfoStatus(true)}>
                                        {pencil}
                                    </Button>
                                }
                                bodyStyle={{paddingTop: 0, paddingBottom: 16}}
                            >
                                <div style={{
                                    width: isMobile ? '100%' : '100%',
                                    float: isMobile ? 'none' : 'left',
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center"
                                }}>
                                    <Descriptions
                                        style={{
                                            display: "flex",
                                            justifyContent: "center",
                                            flexDirection: "column",
                                            alignItems: "center",
                                            width: "60%",
                                            padding: "1rem"
                                        }}
                                    >
                                        <Descriptions.Item label="Full Name" span={3}>
                                            {profileDataValue?.first_name} {profileDataValue?.last_name}
                                        </Descriptions.Item>
                                        {profileDataValue?.index_no && (
                                            <Descriptions.Item label="Index No" span={3}>
                                                {profileDataValue?.index_no}
                                            </Descriptions.Item>
                                        )}
                                        <Descriptions.Item label="Phone numbers" span={3}>
                                            {profileDataValue?.phone_num1}{" "}
                                            <div style={{margin: "0px 10px"}}></div>
                                            {profileDataValue?.phone_num2 || ""}
                                        </Descriptions.Item>
                                        <Descriptions.Item label="Email" span={3}>
                                            {profileDataValue?.email || ""}
                                        </Descriptions.Item>
                                        <Descriptions.Item label="Department" span={3}>
                                            {profileDataValue?.department || ""}
                                        </Descriptions.Item>
                                        <Descriptions.Item label="University" span={3}>
                                            {profileDataValue?.university || ""}
                                        </Descriptions.Item>

                                        {profileDataValue?.year && (
                                            <Descriptions.Item label="Year" span={3}>
                                                {profileData?.year}
                                                {profileData?.performance_level}
                                            </Descriptions.Item>
                                        )}


                                         <Descriptions.Item>

        {profileDataValue &&
        profileDataValue.gpas &&
        profileDataValue.gpas.length ? (
          <Col span={24} md={12} className="mb-24">
            <Card
                style={{display:"none"}}
              bordered={false}
              title={<h6 className="font-semibold m-0">GPA</h6>}
              className="header-solid h-full"
              bodyStyle={{ paddingTop: 0, paddingBottom: 16 }}
            >
              <Row>
                <Col lg={6}>


                    <table>
                      <tbody>
                        <tr>
                          <td>
                            <div style={{ width: "50px" }}></div>
                          </td>
                          <td style={{ textAlign: "left" }}>
                            <span
                              style={{
                                color: "#070C83",
                                fontWeight: 600,
                                fontSize: "15px",
                              }}
                            >
                              Your GPA:
                            </span>
                          </td>
                          <td>
                            <div style={{ width: "80px" }}></div>
                          </td>

                          <td style={{ textAlign: "left", width: "600px" }}>
                            <span>{profileDataValue.gpas[0]?.value}</span>
                          </td>
                        </tr>
                      </tbody>
                    </table>

                    <div style={{ height: "20px" }}></div>
                    <table>
                      <tbody>
                        <tr>
                          <td>
                            <div style={{ width: "40px" }}></div>
                          </td>

                        </tr>
                      </tbody>
                    </table>

                    <div>
                      <table>
                        <tbody>
                          <tr>
                            <td>
                              <div style={{ width: "50px" }}></div>
                            </td>

                            <td style={{ textAlign: "left" }}>
                              <Button
                                 onClick={() => setGpaModal(true)}
                                style={{
                                  background:
                                    "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                  border: "none",
                                  color: "white",
                                  width: "100px",
                                }}
                              >
                                Update
                              </Button>
                            </td>

                            <td>
                              <div style={{ width: "20px" }}></div>
                            </td>

                          </tr>
                        </tbody>
                      </table>

                    </div>


                </Col>
                <Col lg={6}>

                </Col>
              </Row>
              <Modal
                visible={gpaModalStatus}
                footer={null}
                onCancel={() => setGpaModal(false)}
              >
                <GpaEdit score={profileDataValue.gpas[0]?.value} />
              </Modal>


            </Card>
          </Col>
        ) : (
          <></>
        )}</Descriptions.Item>
                                    </Descriptions>
                                </div>


                            </Card>
                        </Col>


                    </Row>


                    <Modal
                        visible={clubModal}
                        footer={null}
                        width={600}
                        onCancel={() => {
                            setClubModal(false);
                            setSelectedClub([]);
                        }}
                    >
                        {" "}
                        <Card bordered={false} className="header-solid mb-24">
                            <Row gutter={[24, 24]}>


                                {selectedClub &&
                                    selectedClub.map((p, index) => (
                                        <Col span={24} md={24} xl={8} key={index}>
                                            {console.log(p, "test")}

                                            <Card bordered={false} className="card-project">
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Club Name
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "90px"}}></div>
                                                        </td>


                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <Input
                                                                onChange={(e) => {
                                                                    {
                                                                        let clubDataVal = [...selectedClub];
                                                                        clubDataVal[index].club_name = e.target.value;
                                                                        setSelectedClub(clubDataVal);
                                                                    }
                                                                }}
                                                                value={p?.club_name}
                                                                style={{width: "300px"}}
                                                                // onChange={(e) => setName(e.target.value)}
                                                                // value={des}
                                                                //   style={{ height: '150px' }}
                                                                // onChange={(e) => setDesc(e.target.value)}
                                                            />
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Position
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "80px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <Select
                                                                style={{width: "300px"}}
                                                                defaultValue="member"
                                                                onChange={(e) => {
                                                                    {
                                                                        let clubDataVal = [...selectedClub];
                                                                        clubDataVal[index].club_level = e;
                                                                        setSelectedClub(clubDataVal);
                                                                    }
                                                                }}
                                                                value={p?.club_level}
                                                                options={[
                                                                    {value: "president", label: "president"},
                                                                    {value: "lead", label: "lead"},
                                                                    {value: "member", label: "member"},
                                                                ]}
                                                            />
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <div style={{height: "20px"}}></div>
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "40px"}}></div>
                                                        </td>


                                                    </tr>
                                                    </tbody>
                                                </table>

                                                <div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            {profileDataValue &&
                                                                selectedClub &&
                                                                !(
                                                                    selectedClub[index].club_name == "" ||
                                                                    selectedClub[index].club_level == ""
                                                                ) && (
                                                                    <td style={{textAlign: "left"}}>
                                                                        <Button
                                                                            onClick={(e) => {
                                                                                updateStu(
                                                                                    {
                                                                                        type: "CLUBS",
                                                                                        club_level: p.club_level,
                                                                                        club_name: p.club_name,
                                                                                        id: p.id,
                                                                                    },
                                                                                    dispatch
                                                                                );
                                                                            }}
                                                                            style={{
                                                                                background:
                                                                                    "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                                border: "none",
                                                                                color: "white",
                                                                                width: "150px",
                                                                            }}
                                                                        >
                                                                            Update
                                                                        </Button>
                                                                    </td>
                                                                )}
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </Card>
                                        </Col>
                                    ))}
                            </Row>
                        </Card>
                    </Modal>

                    <Modal
                        visible={updateProfilePicStatusModal}
                        onOk={() => {
                            if (fileVal.length) {
                                let formData = new FormData();
                                formData.append("file", fileVal[0]?.originFileObj);
                                updateProfilePicDetails(formData, dispatch);
                                setUpdateProfilePicStatus(false);
                            } else {
                                setErrorStaus(true);
                            }
                        }}
                        onCancel={() => setUpdateProfilePicStatus(false)}
                    >
                        <Dragger
                            fileList={fileVal}
                            accept="image/png, image/jpeg"
                            maxCount={1}
                            multiple={false}
                            onChange={handleFile}
                            customRequest={dummyRequest}
                            //   onChange={handleFile}
                        >
                            <p className="ant-upload-drag-icon">
                                <InboxOutlined/>
                            </p>
                            <p
                                className="ant-upload-text"
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "10px",
                                }}
                            >
                                Click or drag file to this area to upload
                            </p>
                        </Dragger>

                        {picErrorStatus && (
                            <span style={{color: "red", marginTop: "10px"}}>
            please select the image
          </span>
                        )}
                    </Modal>
                    <Modal
                        footer={null}
                        onCancel={() => setUpdateInfoStatus(false)}
                        visible={openUpdateProfileInfoStatus}
                    >
                        <UpdateProfile hideModal={hideModal}/>
                    </Modal>
                </div>
                <div style={{
                    width: "60%",
                    display: "flex",
                    justifyContent: "center",
                    flexDirection: "column",
                    gap: "10px"
                }}>

                    {profileDataValue?.skills && profileDataValue.skills.length > 0 && (
                        <div style={{
                            display: "flex",
                            gap: "10px",
                            padding: "2rem",
                            borderRadius: "20px",
                            justifyContent: "center",
                            alignItems: "center",
                            background: "#fff",
                            width: "100%"
                        }}>
                            {profileDataValue?.skills.map((data, index) => {
                                if (
                                    data?.knowledge ||
                                    data?.leadership ||
                                    data?.problem_solving ||
                                    data?.team_work ||
                                    data?.communication ||
                                    data?.creativity ||
                                    data?.performance_level ||
                                    data?.descision_making 
                                    
                                ) {
                                    // const handleClick = () => {
                                    // if (data?.performance_level === 'easy') {
                                    //     alert('This is an easy level performance message.');
                                    // } else if (data?.performance_level === 'better') {
                                    //     alert('This is a better level performance message.');
                                    // } else if (data?.performance_level === 'hard') {
                                    //     alert('This is a hard level performance message.');
                                    // } else {
                                    //     alert('Unknown performance level.');
                                    // }
                                //};
                               


                                    
                                    
                                    const tableCellStyle = {
                                        border: '1px solid #dddddd',
                                        padding: '12px',         // Adjusted padding
                                        textAlign: 'center',     // Center-aligned text
                                        backgroundColor: '#f2f2f2',  // Light gray background color
                                        color: '#333',            // Darker text color
                                    };
                                        
                                    return (
                                        <Space wrap key={index}>
                                             <Descriptions.Item>
                                <Button
                                    onClick={handleClick}
                                    style={{
                                        marginLeft: '10px',
                                        background: 'darkblue', // Replace with your desired gradient colors
                                        border: 'none',
                                        color: 'white',
                                        width: '150px',
                                        transition: 'background 0.3s ease', // Add smooth transition for the background color change
                                       
                                      }}
                                  
                                >
                                    View FeedBack


                                 </Button></Descriptions.Item>
                                
                                 {showModal && (
                <div className="modal">
                    <div className="modal-content">
                        <span className="close" onClick={handleClick}>&times;</span>
                        <p className="modal-message"><b>
                            {data?.performance_level === 'easy' && 'According to your performance, you able to get your internship in an easy way. you are in a correct path, to sustain in this position keep your GPA in a good level. Do Self learning, Hackathon, Sports, Club activities continiously.'}
                            {data?.performance_level === 'better' && (<><b>According to your performance, You are in a moderate way for searching an internship. So that you need to work littlebit to secure your internship in an easy way. for that,  <br /><br /> Activity 1 : Try to keep your GPA above 3.00, to improve your knowledge skill<br />Activity 2 : Do more Selflearning in your intrested area, to improve your knowledge skill<br /> Activity 3 : Participate in club activities
                             to improve your communication, teamwork and leadership skills<br /> Activity 4 : Participate in Hackathon, Designathon or some other events to improve your problem solving skill<br /> Activity 5 : Participate in sports, to improve your decision making, team work and leadership skills.</b></>)}
                            {data?.performance_level === 'hard' && (<><b>According to your performance, You are in a hard way for searching an internship. So that you need to work harder to secure your internship in an easy way. for that,  <br /><br /> Activity 1 : Try to keep your GPA above 3.00, to improve your knowledge skill<br />Activity 2 : Do more Selflearning in your intrested area, to improve your knowledge skill<br /> Activity 3 : Participate in club activities
                             to improve your communication, teamwork and leadership skills<br /> Activity 4 : Participate in Hackathon, Designathon or some other events to improve your problem solving skill<br /> Activity 5 : Participate in sports, to improve your decision making, team work and leadership skills.</b></>)}
                        </b>
                        </p>
                    </div>
                </div>
            )}
                                            {/* <p>dsczzzzzzzzzzzzds: {data?.performance_level}</p> */}
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={'small'} type="circle"
                                                          percent={parseFloat((data.knowledge / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.knowledge} points</b>
                                                <b>Knowledge </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={'small'} type="circle"
                                                          percent={parseFloat((data.leadership / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.leadership} points</b>
                                                <b>Leadership </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={80} type="circle"
                                                          percent={parseFloat((data.problem_solving / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.problem_solving} points</b>
                                                <b>Problem Solving </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={80} type="circle"
                                                          percent={parseFloat((data.team_work / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.team_work} points</b>
                                                <b>Team work </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={80} type="circle"
                                                percent={parseFloat((data.communication / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}
                                                        //   percent={parseFloat((data.communication / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}
                                                        />
                                                <b>{data.communication} points</b>
                                                <b>Communication </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={80} type="circle"
                                                          percent={parseFloat((data.creativity / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.creativity} points</b>
                                                <b>Creativity </b>
                                            </div>
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                textAlign: "center",
                                                gap: "10px"
                                            }}>
                                                <Progress size={80} type="circle"
                                                          percent={parseFloat((data.descision_making / (data.creativity + data.communication + data.team_work + data.problem_solving + data.leadership + data.knowledge + data.descision_making)).toFixed(2)) * 100}/>
                                                <b>{data.descision_making} points</b>
                                                <b>Decision Making </b>
                                                
                                            </div>
                                           
                                        </Space>
                                    );
                                }
                            })}
                        </div>
                    )}


                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#fcfcfc",
                            padding: "1rem",
                        }}>
                        {profileDataValue?.role?.role_code === "STUDENT" &&
                        profileDataValue &&
                        profileDataValue.gpas &&
                        profileDataValue.gpas.length ? (
                            <div style={{width: "80%", display: "flex"}}>
                                <div style={{width: "40%", fontWeight: 600, fontSize: "24px"}}>
                                    GPA
                                </div>
                                <div style={{width: "40%", fontWeight: 600, fontSize: "24px"}}>
                                    {profileDataValue.gpas[0]?.value}
                                </div>
                            </div>

                        ) : (
                            <></>
                        )}


                        {profileDataValue?.role?.role_code === "STUDENT" ? (
                            <Descriptions.Item>
                                <Button
                                    onClick={() => setGpaModal(true)}
                                    style={{
                                        marginLeft: "10px",
                                        background:
                                            "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                        border: "none",
                                        color: "white",
                                        width: "100px",
                                    }}
                                >
                                    Update GPA


                                </Button></Descriptions.Item>
                        ) : (
                            <></>
                        )}
                    </div>





                    {selfLearningData && selfLearningData?.length ? (
                        <Card
                            style={{background:"#E6E7FF",width:"100%"}}
                            bordered={false}
                            className="header-solid mb-24"
                            title={
                                <>
                                    <h6 className="font-semibold">Self Learning</h6>
                                    {/* <p>Architects design houses</p> */}
                                </>
                            }
                        >
                            <div style={{display:"flex",gap:"10px",alignItems:"center",justifyContent:"center",flexWrap:"wrap"}}>


                                {selfLearningData &&
                                    selfLearningData.map((p, index) => (
                                        <>
                                        <div style={{display:"flex",justifyContent:"center",alignItems:"center"}}>
                                                <Card
                                                    key={index}
                                                    bordered={true}
                                                    className="card-project"
                                                    style={{border:"1px solid",padding:"1rem",width:"300px"}}>
                                                    <div style={{display:"flex",flexDirection:"column",gap:"10px"}}>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <div
                                                            style={{
                                                                width:"60%",
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                            }}
                                                        >
                                                          Course Name
                                                        </div>
                                                            <div style={{textAlign: "left"}}>
                                                                <span>{p?.course_name}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <span
                                                            style={{
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                                width:"60%"
                                                            }}
                                                        >
                                                          Level
                                                        </span>
                                                            <div style={{textAlign: "left"}}>
                                                                <span>{p?.course_level}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{textAlign: "left",display:"flex",width:"100%",gap:"10px"}}>
                                                            <Button
                                                                onClick={(e) => {
                                                                    let selfLearning = profileData?.selflearnings
                                                                        ? profileData.selflearnings.map((club) => ({
                                                                            ...club,
                                                                        }))
                                                                        : [];

                                                                    let arry = [];
                                                                    arry.push(selfLearning[index]);
                                                                    serSelectedSelfLearning(arry);
                                                                    setSelfModal(true);
                                                                }}
                                                                style={{
                                                                    background:
                                                                        "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                    border: "none",
                                                                    color: "white",
                                                                }}
                                                            >
                                                                Update
                                                            </Button>
                                                            <Button
                                                                onClick={() =>
                                                                    deleteStu(
                                                                        {
                                                                            type: "SELF_LEARNINGS",
                                                                            id: p.id,
                                                                        },
                                                                        dispatch
                                                                    )
                                                                }
                                                            >
                                                                Remove
                                                            </Button>
                                                        </div>
                                                    </div>
                                                </Card>
                                        </div>
                                        </>

                                    ))}
                            </div>
                        </Card>
                    ) : (
                        <></>
                    )}

                    {sportData && sportData?.length ? (
                        <Card
                            style={{background:"#E6E7FF",width:"100%"}}
                            bordered={false}
                            className="header-solid mb-24"
                            title={
                                <>
                                    <h6 className="font-semibold">Sports</h6>
                                </>
                            }
                        >
                            <Row gutter={[12, 12]}>
                                {sportData &&
                                    sportData.map((p, index) => (
                                        <>
                                            <Col key={index} style={{width:"100%"}}>
                                                <Card bordered={true} className="card-project" style={{border:"1px solid",padding:"1rem"}}>
                                                    <div style={{display:"flex",flexDirection:"column",gap:"10px"}}>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <span
                                                            style={{
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                            }}
                                                        >
                                                          Name
                                                        </span>
                                                            <div style={{textAlign: "left", width: "600px"}}>
                                                                <span>{p?.sport_name}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <span
                                                            style={{
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                            }}
                                                        >
                                                          Level
                                                        </span>
                                                            <div style={{textAlign: "left", width: "600px"}}>
                                                                <span>{p?.level}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <span
                                                            style={{
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                            }}
                                                        >
                                                          position
                                                        </span>

                                                            <div style={{textAlign: "left", width: "600px"}}>
                                                                <span>{p?.position}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{display: "flex", gap: "10px"}}>
                                                        <span
                                                            style={{
                                                                color: "#070C83",
                                                                fontWeight: 600,
                                                                fontSize: "15px",
                                                            }}
                                                        >
                                                          Cardinality
                                                        </span>
                                                            <div style={{textAlign: "left", width: "600px"}}>
                                                                <span>{p?.cardinality}</span>
                                                            </div>
                                                        </div>
                                                        <div style={{textAlign: "left",display:"flex",width:"100%",gap:"10px"}}>
                                                            <Button
                                                                onClick={(e) => {
                                                                    let arry = profileData?.sports
                                                                        ? profileData?.sports.map((club) => ({
                                                                            ...club,
                                                                        }))
                                                                        : [];
                                                                    console.log(arry[index], "ll");

                                                                    let arrNew = [];
                                                                    arrNew.push(arry[index]);
                                                                    console.log(arrNew, "ll");
                                                                    setSelectedSport(arrNew);
                                                                    setSportModal(true);
                                                                }}
                                                                style={{
                                                                    background:
                                                                        "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                    border: "none",
                                                                    color: "white",
                                                                    width: "100px",
                                                                }}
                                                            >
                                                                Update
                                                            </Button>
                                                            <Button
                                                                onClick={() =>
                                                                    deleteStu(
                                                                        {
                                                                            type: "SPORT",
                                                                            id: p.id,
                                                                        },
                                                                        dispatch
                                                                    )
                                                                }
                                                            >
                                                                Remove
                                                            </Button>
                                                        </div>
                                                    </div>
                                                </Card>
                                            </Col>
                                        </>

                                    ))}
                            </Row>
                        </Card>
                    ) : (
                        <></>
                    )}

                    {selectedEventData && selectedEventData?.length ? (
                        <Modal
                            footer={null}
                            visible={eventModal}
                            onCancel={() => {
                                setSelectedEvent([]);
                                setEventModal(false);
                            }}
                            width={600}
                        >
                            <Card
                                bordered={false}
                                className="header-solid mb-24"
                                title={
                                    <>
                                        <h6 className="font-semibold">Events</h6>
                                    </>
                                }
                            >
                                <Row gutter={[24, 24]}>

                                    {selectedEventData &&
                                        selectedEventData.map((p, index) => (
                                            <Col span={24} md={24} xl={8} key={index}>
                                                <Card bordered={false} className="card-project">
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Name
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "90px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Input
                                                                    value={p?.event_name}
                                                                    style={{width: "300px"}}
                                                                    onChange={(e) => {
                                                                        let clubDataVal = [...selectedEventData];
                                                                        clubDataVal[index].event_name =
                                                                            e.target.value;
                                                                        setSelectedEvent(clubDataVal);
                                                                    }}
                                                                    // value={des}
                                                                    //   style={{ height: '150px' }}
                                                                    // onChange={(e) => setDesc(e.target.value)}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Level
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "100px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    style={{width: "300px"}}
                                                                    defaultValue="university"
                                                                    value={p?.level}
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedEventData];
                                                                            clubDataVal[index].level = e;
                                                                            setSelectedEvent(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "university",
                                                                            label: "university level",
                                                                        },
                                                                        {
                                                                            value: "national",
                                                                            label: "national level",
                                                                        },
                                                                        {
                                                                            value: "international",
                                                                            label: "international level",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                position
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "80px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    value={p?.position}
                                                                    style={{width: "300px"}}
                                                                    defaultValue="1"
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedEventData];
                                                                            clubDataVal[index].position = e;
                                                                            setSelectedEvent(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "1",
                                                                            label: "1",
                                                                        },
                                                                        {value: "2", label: "2"},
                                                                        {
                                                                            value: "3",
                                                                            label: "3",
                                                                        },
                                                                        {
                                                                            value: "part",
                                                                            label: "part",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "30px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Cardinality
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "80px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    style={{width: "300px"}}
                                                                    value={p?.cardinality}
                                                                    defaultValue="1"
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedEventData];
                                                                            clubDataVal[index].cardinality = e;
                                                                            setSelectedEvent(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "leader",
                                                                            label: "leader",
                                                                        },
                                                                        {value: "member", label: "member"},
                                                                        {
                                                                            value: "individual",
                                                                            label: "individual",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>

                                                    <div style={{height: "20px"}}></div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "40px"}}></div>
                                                            </td>

                                                        </tr>
                                                        </tbody>
                                                    </table>

                                                    <div>
                                                        <table>
                                                            <tbody>
                                                            <tr>
                                                                <td>
                                                                    <div style={{width: "120px"}}></div>
                                                                </td>
                                                                {selectedEventData[index].event_name != "" &&
                                                                    selectedEventData[index].level != "" && (
                                                                        <td style={{textAlign: "left"}}>
                                                                            <Button
                                                                                onClick={(e) => {
                                                                                    updateStu(
                                                                                        {
                                                                                            type: "EVENT",
                                                                                            event_name: p.event_name,
                                                                                            level: p.level,
                                                                                            position: p.position,
                                                                                            cardinality: p.cardinality,
                                                                                            id: p.id,
                                                                                        },
                                                                                        dispatch
                                                                                    );
                                                                                }}
                                                                                style={{
                                                                                    background:
                                                                                        "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                                    border: "none",
                                                                                    color: "white",
                                                                                    width: "100px",
                                                                                }}
                                                                            >
                                                                                Update
                                                                            </Button>
                                                                        </td>
                                                                    )}
                                                            </tr>
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </Card>
                                            </Col>
                                        ))}
                                </Row>
                            </Card>
                        </Modal>
                    ) : (
                        <></>
                    )}

                    {clubData && clubData?.length ? (
                        <Card
                            style={{background:"#E6E7FF",width:"100%"}}
                            bordered={false}
                            className="header-solid mb-24"
                            title={
                                <>
                                    <h6 className="font-semibold">Clubs</h6>
                                    {/* <p>Architects design houses</p> */}
                                </>
                            }
                        >
                            <Row style={{width:"100%"}}>


                                {clubData &&
                                    clubData.map((p, index) => (
                                        <div style={{width:"100%",display:"flex",justifyContent:"center"}} key={index}>

                                            <Card
                                                style={{padding:"0.5rem",width:"100%",border:"1px solid black"}}
                                                bordered={true} className="card-project">
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Club Name
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "90px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p.club_name}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Position
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "80px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p.club_level}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <div style={{height: "20px"}}></div>
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "40px"}}></div>
                                                        </td>

                                                    </tr>
                                                    </tbody>
                                                </table>
                                                {console.log(
                                                    profileDataValue.clubs[index]?.club_name,
                                                    clubData[index].club_name
                                                )}

                                                <div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left"}}>
                                                                <Button
                                                                    onClick={(e) => {
                                                                        let clubArray = profileData?.clubs
                                                                            ? profileData.clubs.map((club) => ({
                                                                                ...club,
                                                                            }))
                                                                            : [];
                                                                        let arr = clubArray[index];
                                                                        let arrayval = [];
                                                                        arrayval.push(arr);
                                                                        console.log(arrayval, "arr");
                                                                        setSelectedClub(arrayval);

                                                                        setClubModal(true);
                                                                    }}
                                                                    style={{
                                                                        background:
                                                                            "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                        border: "none",
                                                                        color: "white",
                                                                        width: "100px",
                                                                    }}
                                                                >
                                                                    Update
                                                                </Button>
                                                            </td>

                                                            <td>
                                                                <div style={{width: "20px"}}></div>
                                                            </td>
                                                            <td>
                                                                <Button
                                                                    onClick={() =>
                                                                        deleteStu(
                                                                            {
                                                                                type: "CLUBS",
                                                                                id: p.id,
                                                                            },
                                                                            dispatch
                                                                        )
                                                                    }
                                                                >
                                                                    Remove
                                                                </Button>
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </Card>
                                        </div>
                                    ))}
                            </Row>
                        </Card>
                    ) : (
                        <></>
                    )}



                    {selectedSportData && selectedSportData?.length ? (
                        <Modal
                            visible={sportModal}
                            footer={null}
                            width={600}
                            onCancel={() => {
                                setSportModal(false);
                                setSelectedSport([]);
                            }}
                        >
                            <Card
                                bordered={false}
                                className="header-solid mb-24"
                                title={
                                    <>
                                        <h6 className="font-semibold">Sports</h6>
                                        {/* <p>Architects design houses</p> */}
                                    </>
                                }
                            >
                                <Row gutter={[24, 24]}>


                                    {selectedSportData &&
                                        selectedSportData.map((p, index) => (
                                            <Col span={24} md={24} xl={8} key={index}>
                                                <Card
                                                    style={{padding:"1rem"}}
                                                    bordered={false} className="card-project">
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Name
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "90px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Input
                                                                    value={p?.sport_name}
                                                                    style={{width: "300px"}}
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedSportData];
                                                                            clubDataVal[index].sport_name =
                                                                                e.target.value;

                                                                            setSelectedSport(clubDataVal);
                                                                        }
                                                                    }}
                                                                    // value={des}
                                                                    //   style={{ height: '150px' }}
                                                                    // onChange={(e) => setDesc(e.target.value)}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Level
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "100px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    style={{width: "300px"}}
                                                                    defaultValue="university"
                                                                    value={p?.level}
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedSportData];
                                                                            clubDataVal[index].level = e;

                                                                            setSelectedSport(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "university",
                                                                            label: "university level",
                                                                        },
                                                                        {
                                                                            value: "national",
                                                                            label: "national level",
                                                                        },
                                                                        {
                                                                            value: "international",
                                                                            label: "international level",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                position
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "80px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    value={p?.position}
                                                                    style={{width: "300px"}}
                                                                    defaultValue="1"
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedSportData];
                                                                            clubDataVal[index].position = e;

                                                                            setSelectedSport(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "1",
                                                                            label: "1",
                                                                        },
                                                                        {value: "2", label: "2"},
                                                                        {
                                                                            value: "3",
                                                                            label: "3",
                                                                        },
                                                                        {
                                                                            value: "part",
                                                                            label: "part",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "30px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Cardinality
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "80px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    style={{width: "300px"}}
                                                                    value={p?.cardinality}
                                                                    defaultValue="1"
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...selectedSportData];
                                                                            clubDataVal[index].cardinality = e;

                                                                            setSelectedSport(clubDataVal);
                                                                        }
                                                                    }}
                                                                    options={[
                                                                        {
                                                                            value: "leader",
                                                                            label: "leader",
                                                                        },
                                                                        {value: "member", label: "member"},
                                                                        {
                                                                            value: "individual",
                                                                            label: "individual",
                                                                        },
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>

                                                    <div style={{height: "20px"}}></div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "40px"}}></div>
                                                            </td>

                                                        </tr>
                                                        </tbody>
                                                    </table>

                                                    <div>
                                                        <table>
                                                            <tbody>
                                                            <tr>
                                                                <td>
                                                                    <div style={{width: "150px"}}></div>
                                                                </td>

                                                                {selectedSportData[index].sport_name != "" &&
                                                                    selectedSportData[index].level != "" && (
                                                                        <td style={{textAlign: "left"}}>
                                                                            <Button
                                                                                onClick={(e) => {
                                                                                    updateStu(
                                                                                        {
                                                                                            type: "SPORT",
                                                                                            event_name: p.sport_name,
                                                                                            level: p.level,
                                                                                            position: p.position,
                                                                                            cardinality: p.cardinality,
                                                                                            id: p.id,
                                                                                        },
                                                                                        dispatch
                                                                                    );
                                                                                }}
                                                                                style={{
                                                                                    background:
                                                                                        "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                                    border: "none",
                                                                                    color: "white",
                                                                                    width: "100px",
                                                                                }}
                                                                            >
                                                                                Update
                                                                            </Button>
                                                                        </td>
                                                                    )}
                                                            </tr>
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </Card>
                                            </Col>
                                        ))}
                                </Row>
                            </Card>
                        </Modal>
                    ) : (
                        <></>
                    )}


                    {SelectedSelfLearning && SelectedSelfLearning?.length ? (
                        <Modal
                            footer={null}
                            width={600}
                            visible={selfModal}
                            onCancel={() => {
                                setSelfModal(false);
                                serSelectedSelfLearning([]);
                            }}
                        >
                            <Card
                                bordered={false}
                                className="header-solid mb-24"
                                title={
                                    <>
                                        <h6 className="font-semibold">Self Learning</h6>
                                        {/* <p>Architects design houses</p> */}
                                    </>
                                }
                            >
                                <Row gutter={[24, 24]}>
                                    {/* {project.map((p, index) => (
            <Col span={24} md={12} xl={6} key={index}>
              <Card
                bordered={false}
                className="card-project"
                cover={<img alt="example" src={p.img} />}
              >
                <div className="card-tag">{p.titlesub}</div>
                <h5>{p.titile}</h5>
                <p>{p.disciption}</p>
                <Row gutter={[6, 0]} className="card-footer">
                  <Col span={12}>
                    <Button type="button">VIEW PROJECT</Button>
                  </Col>
                  <Col span={12} className="text-right">
                    <Avatar.Group className="avatar-chips">
                      <Avatar size="small" src={profilavatar} />
                      <Avatar size="small" src={convesionImg} />
                      <Avatar size="small" src={convesionImg2} />
                      <Avatar size="small" src={convesionImg3} />
                    </Avatar.Group>
                  </Col>
                </Row>
              </Card>
            </Col>
          ))} */}

                                    {SelectedSelfLearning &&
                                        SelectedSelfLearning.map((p, index) => (
                                            <Col span={24} md={24} xl={8} key={index}>
                                                {console.log(p, "test")}

                                                <Card bordered={false} className="card-project">
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Course Name
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "90px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Input
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...SelectedSelfLearning];
                                                                            clubDataVal[index].course_name =
                                                                                e.target.value;
                                                                            serSelectedSelfLearning(clubDataVal);
                                                                        }
                                                                    }}
                                                                    value={p?.course_name}
                                                                    style={{width: "300px"}}
                                                                    // onChange={(e) => setName(e.target.value)}
                                                                    // value={des}
                                                                    //   style={{ height: '150px' }}
                                                                    // onChange={(e) => setDesc(e.target.value)}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>
                                                            <td style={{textAlign: "left"}}>
                              <span
                                  style={{
                                      color: "#070C83",
                                      fontWeight: 600,
                                      fontSize: "15px",
                                  }}
                              >
                                Position
                              </span>
                                                            </td>
                                                            <td>
                                                                <div style={{width: "80px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left", width: "600px"}}>
                                                                <Select
                                                                    style={{width: "300px"}}
                                                                    defaultValue="member"
                                                                    onChange={(e) => {
                                                                        {
                                                                            let clubDataVal = [...SelectedSelfLearning];
                                                                            clubDataVal[index].course_level = e;
                                                                            serSelectedSelfLearning(clubDataVal);
                                                                        }
                                                                    }}
                                                                    value={p?.course_level}
                                                                    options={[
                                                                        {value: "beginner", label: "beginner"},
                                                                        {
                                                                            value: "intermediate",
                                                                            label: "intermediate",
                                                                        },
                                                                        {value: "expert", label: "expert"},
                                                                    ]}
                                                                />
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                    <div style={{height: "20px"}}></div>

                                                    <div style={{height: "20px"}}></div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "40px"}}></div>
                                                            </td>
                                                            {/* <td style={{ textAlign: 'left' }}>
                            <span
                              style={{
                                color: '#070C83',
                                fontWeight: 600,
                                fontSize: '15px',
                              }}
                            >
                              Select a File
                            </span>
                          </td> */}
                                                            {/* <td>
                            <div style={{ width: '90px' }}></div>
                          </td> */}

                                                            {/* <td style={{ textAlign: 'left', width: '600px' }}>
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
                          </td> */}
                                                        </tr>
                                                        </tbody>
                                                    </table>

                                                    <div>
                                                        <table>
                                                            <tbody>
                                                            <tr>
                                                                <td>
                                                                    <div style={{width: "150px"}}></div>
                                                                </td>
                                                                {SelectedSelfLearning[index].course_name != "" &&
                                                                    SelectedSelfLearning[index].course_level !=
                                                                    "" && (
                                                                        <td style={{textAlign: "left"}}>
                                                                            <Button
                                                                                onClick={(e) => {
                                                                                    updateStu(
                                                                                        {
                                                                                            type: "SELF_LEARNINGS",
                                                                                            course_level: p.course_level,
                                                                                            course_name: p.course_name,
                                                                                            id: p.id,
                                                                                        },
                                                                                        dispatch
                                                                                    );
                                                                                }}
                                                                                style={{
                                                                                    background:
                                                                                        "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                                    border: "none",
                                                                                    color: "white",
                                                                                    width: "100px",
                                                                                }}
                                                                            >
                                                                                Update
                                                                            </Button>
                                                                        </td>
                                                                    )}
                                                            </tr>
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </Card>
                                            </Col>
                                        ))}
                                </Row>
                            </Card>
                        </Modal>
                    ) : (
                        <></>
                    )}

                    {eventData && eventData?.length ? (
                        <Card
                            style={{background:"#E6E7FF"}}
                            bordered={false}
                            className="header-solid mb-24"
                            title={
                                <>
                                    <h6 className="font-semibold">Events</h6>
                                    {/* <p>Architects design houses</p> */}
                                </>
                            }
                        >
                            <Row gutter={[24, 24]}
                                 style={{display: "flex", alignItems: "center", flexDirection: "column"}}>


                                {eventData &&
                                    eventData.map((p, index) => (
                                        <Col style={{width:"100%",padding:"1rem"}} key={index}>
                                            <Card
                                                style={{padding:"1rem",border:"1px solid"}}
                                                bordered={true} className="card-project">
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Name
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "90px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p?.event_name}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Level
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "100px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p?.level}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "50px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              position
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "80px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p?.position}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                                <div style={{height: "20px"}}></div>

                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "30px"}}></div>
                                                        </td>
                                                        <td style={{textAlign: "left"}}>
                            <span
                                style={{
                                    color: "#070C83",
                                    fontWeight: 600,
                                    fontSize: "15px",
                                }}
                            >
                              Cardinality
                            </span>
                                                        </td>
                                                        <td>
                                                            <div style={{width: "80px"}}></div>
                                                        </td>

                                                        <td style={{textAlign: "left", width: "600px"}}>
                                                            <span>{p?.cardinality}</span>
                                                        </td>
                                                    </tr>
                                                    </tbody>
                                                </table>

                                                <div style={{height: "20px"}}></div>
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style={{width: "40px"}}></div>
                                                        </td>

                                                    </tr>
                                                    </tbody>
                                                </table>

                                                <div>
                                                    <table>
                                                        <tbody>
                                                        <tr>
                                                            <td>
                                                                <div style={{width: "50px"}}></div>
                                                            </td>

                                                            <td style={{textAlign: "left"}}>
                                                                <Button
                                                                    onClick={(e) => {
                                                                        let events = profileData?.events
                                                                            ? profileData.events.map((club) => ({
                                                                                ...club,
                                                                            }))
                                                                            : [];
                                                                        let arr = [];
                                                                        arr.push(events[index]);
                                                                        setSelectedEvent(arr);
                                                                        setEventModal(true);
                                                                    }}
                                                                    style={{
                                                                        background:
                                                                            "linear-gradient(to right, #000AF3, #69D0F0)", // Replace with your desired gradient colors
                                                                        border: "none",
                                                                        color: "white",
                                                                        width: "100px",
                                                                    }}
                                                                >
                                                                    Update
                                                                </Button>
                                                            </td>

                                                            <td>
                                                                <div style={{width: "20px"}}></div>
                                                            </td>
                                                            <td>
                                                                <Button
                                                                    onClick={() =>
                                                                        deleteStu(
                                                                            {
                                                                                type: "EVENT",
                                                                                id: p.id,
                                                                            },
                                                                            dispatch
                                                                        )
                                                                    }
                                                                >
                                                                    Remove
                                                                </Button>
                                                            </td>
                                                        </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </Card>
                                        </Col>
                                    ))}
                            </Row>

                        </Card>
                    ) : (
                        <></>
                    )}
                </div>
            </div>

        </>
    );
}

export default Profile;
