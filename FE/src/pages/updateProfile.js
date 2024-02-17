import React, { useEffect, useState } from 'react';
import { Button, Col, Form, Input, Row, Tabs } from 'antd';
import { useDispatch, useSelector } from 'react-redux';

import { signUp, updateProfileDetails } from '../redux/action';
import { useForm } from 'antd/es/form/Form';

const UpdateProfile = ({ hideModal }) => {
  const role = useSelector((state) => state?.commonReducer?.roleCode);
  const profileData = useSelector(
    (state) => state?.studentReducer?.profileData
  );

  const [tab, setTab] = useState('1');
  const dispatch = useDispatch();
  const [form1] = useForm();
  const [form2] = useForm();

  useEffect(() => {
    if (role === 'STUDENT') {
      setTab('1');
    } else {
      setTab('2');
    }
  }, []);

  //   useEffect(() => {
  //     if (isSignup === true) {
  //       Swal.fire({
  //         title:
  //           'Successfully Registered Please Check your inbox for activation email ',
  //         // text: "You won't be able to revert this!",
  //         icon: 'success',
  //         showCancelButton: false,
  //         confirmButtonColor: '#3085d6',
  //         cancelButtonColor: '#d33',
  //         confirmButtonText: 'Login',
  //       }).then((result) => {
  //         if (result.isConfirmed) {
  //           history.push('/login');
  //         }
  //       });
  //     }
  //     if (isSignup === false) {
  //       Swal.fire({
  //         title: 'Error Occurred while registering ',
  //         // text: "You won't be able to revert this!",
  //         icon: 'error',
  //         showCancelButton: false,
  //         confirmButtonColor: '#3085d6',
  //         cancelButtonColor: '#d33',
  //         confirmButtonText: 'OK',
  //       });
  //     }
  //   }, [isSignup]);

  const onFinish = (values) => {
    updateProfileDetails(values, dispatch);
    hideModal();
  };

  const handleTabs = (val) => {
    setTab(val);
  };

  return (
    <Tabs activeKey={tab} onChange={handleTabs}>
      <Tabs.TabPane
        tab="Update Student Details"
        key="1"
        disabled={role !== 'STUDENT'}
      >
        <Form
          form={form1}
          name="basic"
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          style={{ maxWidth: 600 }}
          initialValues={{
            first_name: profileData?.first_name,
            last_name: profileData?.last_name,
            index_no: profileData?.index_no,
            year: profileData?.year,
            department: profileData?.department,
            university: profileData?.university,
            phone_num1: profileData?.phone_num1,
            phone_num2: profileData?.phone_num2,
          }}
          onFinish={onFinish}
          autoComplete="off"
        >
          <table>
            <tbody>
              <tr>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="first_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your first name',
                      },
                    ]}
                  >
                    <Input
                      placeholder="First Name"
                      defaultValue={profileData?.first_name || ''}
                    />
                  </Form.Item>
                </td>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="last_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your first name',
                      },
                    ]}
                  >
                    <Input
                      placeholder="Last Name"
                      defaultValue={profileData?.last_name || ''}
                    />
                  </Form.Item>
                </td>
              </tr>
            </tbody>
          </table>

          <table>
            <tbody>
              <tr>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="index_no"
                    rules={[
                      { required: true, message: 'Please input your Index No' },
                    ]}
                  >
                    <Input
                      placeholder="Index No"
                      defaultValue={profileData?.index_no || ''}
                    />
                  </Form.Item>
                </td>
                <td>
                  <Form.Item
                    name="year"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your academic year',
                      },
                    ]}
                  >
                    <Input
                      placeholder="Academic Year"
                      defaultValue={profileData?.year || ''}
                    />
                  </Form.Item>
                </td>
              </tr>
            </tbody>
          </table>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="department"
                rules={[
                  { required: true, message: 'Please input your Department' },
                ]}
              >
                <Input
                  placeholder="Department"
                  style={{ width: '330px' }}
                  defaultValue={profileData?.department || ''}
                />
              </Form.Item>
            </Col>
          </Row>

          <table>
            <tbody>
              <tr>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="phone_num1"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your phone number',
                      },
                    ]}
                  >
                    <Input
                      placeholder="Phone number1"
                      defaultValue={profileData?.phone_num1 || ''}
                    />
                  </Form.Item>
                </td>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="phone_num2"
                    // rules={[
                    //   { required: true, message: 'Please input your first name' },
                    // ]}
                  >
                    <Input
                      placeholder="Phone number2"
                      defaultValue={profileData?.phone_num2 || ''}
                    />
                  </Form.Item>
                </td>
              </tr>
            </tbody>
          </table>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="university"
                rules={[
                  { required: true, message: 'Please input your university' },
                ]}
              >
                <Input
                  placeholder="university"
                  style={{ width: '330px' }}
                  defaultValue={profileData?.university || ''}
                />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ marginTop: '10px' }}></div>
          <Row>
            {/* <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="password_2"
                rules={[
                  { required: true, message: 'Please input your Password' },
                ]}
              >
                <Input placeholder="Password" style={{ width: '330px' }} />
              </Form.Item>
            </Col> */}
          </Row>
          <Row>
            {/* <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="confirm_password_2"
                rules={[{ required: true, message: 'Please Confirm Password' }]}
              >
                <Input
                  placeholder="Confirm Password"
                  style={{ width: '330px' }}
                />
              </Form.Item>
            </Col> */}
          </Row>
          {/* <Row>
            {isPassError && (
              <span style={{ color: 'red' }}>Password does not match</span>
            )}
          </Row> */}

          <Row>
            <Col lg={24}>
              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  style={{ width: '330px' }}
                >
                  Update
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Tabs.TabPane>
      <Tabs.TabPane
        tab="Update Career Guidence Officer Details"
        key="2"
        disabled={role !== 'CAREER_GUIDANCE_OFFICER'}
      >
        {/* {isPassError && (
          <span style={{ color: 'red' }}>Password does not match</span>
        )} */}
        <Form
          name="basic"
          form={form2}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          style={{ maxWidth: 600 }}
          initialValues={{
            first_name: profileData?.first_name,
            last_name: profileData?.last_name,
            department: profileData?.department,
            university: profileData?.university,
            phone_num1: profileData?.phone_num1,
            phone_num2: profileData?.phone_num2,
          }}
          onFinish={onFinish}
          autoComplete="off"
        >
          <table>
            <tbody>
              <tr>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="first_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your first name',
                      },
                    ]}
                  >
                    <Input
                      placeholder="First Name"
                      defaultValue={profileData?.first_name || ''}
                    />
                  </Form.Item>
                </td>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="last_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your last name',
                      },
                    ]}
                  >
                    <Input
                      placeholder="Last Name"
                      defaultValue={profileData?.last_name || ''}
                    />
                  </Form.Item>
                </td>
              </tr>
            </tbody>
          </table>

          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="department"
                rules={[
                  { required: true, message: 'Please input your Department' },
                ]}
              >
                <Input
                  placeholder="Department"
                  style={{ width: '330px' }}
                  defaultValue={profileData?.department || ''}
                />
              </Form.Item>
            </Col>
          </Row>

          <table>
            <tbody>
              <tr>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="phone_num1"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your phone number',
                      },
                    ]}
                  >
                    <Input
                      placeholder="Phone number1"
                      defaultValue={profileData?.phone_num1 || ''}
                    />
                  </Form.Item>
                </td>
                <td>
                  <Form.Item
                    // label="First Name"
                    name="phone_num2"
                    // rules={[
                    //   { required: true, message: 'Please input your first name' },
                    // ]}
                  >
                    <Input
                      placeholder="Phone number2"
                      defaultValue={profileData?.phone_num2 || ''}
                    />
                  </Form.Item>
                </td>
              </tr>
            </tbody>
          </table>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="university"
                rules={[
                  { required: true, message: 'Please input your university' },
                ]}
              >
                <Input
                  placeholder="university"
                  style={{ width: '330px' }}
                  defaultValue={profileData?.university || ''}
                />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ marginTop: '10px' }}></div>
          {/* <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="password"
                rules={[
                  { required: true, message: 'Please input your Password' },
                ]}
              >
                <Input placeholder="Password" style={{ width: '330px' }} />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            {isPassError && (
              <span style={{ color: 'red' }}>Password does not match</span>
            )}
          </Row> */}
          {/* <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="confirm_password"
                rules={[{ required: true, message: 'Please Confirm Password' }]}
              >
                <Input
                  placeholder="Confirm Password"
                  style={{ width: '330px' }}
                />
              </Form.Item>
            </Col>
          </Row> */}

          <Row>
            <Col lg={24}>
              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  style={{ width: '330px' }}
                >
                  Update
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Tabs.TabPane>
    </Tabs>
  );
};

export default UpdateProfile;
