import React, { useEffect, useState } from 'react';
import { Button, Col, Form, Input, Row, Tabs } from 'antd';
import { useDispatch, useSelector } from 'react-redux';
import Swal from 'sweetalert2';
import { InboxOutlined } from '@ant-design/icons';

import { defaultSignUp, signUp } from '../redux/action';
import { useForm } from 'antd/es/form/Form';
import { useHistory } from 'react-router-dom/cjs/react-router-dom.min';
import Dragger from 'antd/lib/upload/Dragger';

const SignUp = ({ handleTab }) => {
  const isSignup = useSelector((state) => state?.commonReducer?.isSignup);
  const signupMessage = useSelector(
    (state) => state?.commonReducer?.signupMessage
  );

  const [fileVal, setFile] = useState([]);
  const [fileVal2, setFile2] = useState([]);
  const [tab, setTab] = useState('1');
  const [isPassError, setPassError] = useState(false);
  const dispatch = useDispatch();
  const [form1] = useForm();
  const [form2] = useForm();
  const history = useHistory();
  const dummyRequest1 = ({ file, onSuccess }) => {
    setTimeout(() => {
      onSuccess('ok');
    }, 0);
  };

  const dummyRequest2 = ({ file, onSuccess }) => {
    setTimeout(() => {
      onSuccess('ok');
    }, 0);
  };

  useEffect(() => {
    defaultSignUp(dispatch);
  }, []);

  useEffect(() => {
    if (isSignup === true) {
      Swal.fire({
        title:
          'Successfully Registered Please Check your inbox for activation email ',
        // text: "You won't be able to revert this!",
        icon: 'success',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Login',
      }).then((result) => {
        if (result.isConfirmed) {
          history.push('/login');
        }
      });
      defaultSignUp(dispatch);
    }
    if (isSignup === false) {
      Swal.fire({
        title: `${signupMessage}`,
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'OK',
      });
      defaultSignUp(dispatch);
    }
  }, [isSignup]);

  const onFinish = (values) => {
    if (tab == 1) {
      if (values.password_2 !== values.confirm_password_2) {
        setPassError(true);
      } else {
        setPassError(false);

        values = {
          ...values,
          password: values.password_2,
          role_id: parseInt(tab),
        };

        const formData = new FormData();
        formData.append('email', values.email);
        formData.append('phone_num1', values.phone_num1);
        formData.append('phone_num2', values.phone_num2);
        formData.append('first_name', values.first_name);
        formData.append('last_name', values.last_name);
        formData.append('department', values.department);
        formData.append('index_no', values.index_no);
        formData.append('year', values.year);
        formData.append('password', values.password);
        formData.append('university', values.university);
        formData.append('role_id', values.role_id);
        formData.append('file', fileVal[0]?.originFileObj);
        signUp(values, dispatch);
        form1.resetFields();
        setFile([]);
      }
    } else if (tab == 2) {
      if (values.password !== values.confirm_password) {
        setPassError(true);
      } else {
        setPassError(false);
        values = {
          ...values,
          password: values.password,
          role_id: parseInt(tab),
        };
        const formData = new FormData();
        formData.append('email', values.email);
        formData.append('phone_num1', values.phone_num1);
        formData.append('phone_num2', values.phone_num2);
        formData.append('department', values.department);
        formData.append('first_name', values.first_name);
        formData.append('last_name', values.last_name);
        formData.append('password', values.password);
        formData.append('university', values.university);
        formData.append('role_id', values.role_id);
        formData.append('file', fileVal2[0]?.originFileObj);
        signUp(formData, dispatch);
        setFile2([]);
        form2.resetFields();
      }
    }
  };

  const handleTabs = (val) => {
    setTab(val);
  };

  const handleFile = ({ fileList }) => {
    console.log(fileList, tab);

    setFile([...fileList]);
  };

  const handleFile2 = ({ fileList }) => {
    setFile2([...fileList]);
  };

  console.log(fileVal, 'valll');

  return (
    
    <Tabs defaultActiveKey="1" onChange={handleTabs}>
      <Tabs.TabPane tab="Student" key="1">
        <Form
          form={form1}
          name="basic"
          labelCol={{ span: 8 }}
          // wrapperCol={{ span: 16 }}
          // initialValues={{ remember: true }}
          onFinish={onFinish}
          autoComplete="off"
        >
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center',gap:'10px' }}>
                  <Form.Item
                      style={{ width: '100%' }}
                    name="first_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your first name',
                      },
                    ]}
                  >
                    <Input placeholder="First Name" style={{ width: '100%' }} />
                  </Form.Item>
                  <Form.Item
                      style={{ width: '100%' }}
                    name="last_name"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your last name',
                      },
                    ]}
                  >
                    <Input placeholder="Last Name" />
                  </Form.Item>
          </div>

          <div style={{ width: '100%', display: 'flex', justifyContent: 'center',gap:'10px' }}>
                  <Form.Item
                      style={{ width: '100%' }}
                    // label="First Name"
                    name="index_no"
                    rules={[
                      { required: true, message: 'Please input your Index No' },
                    ]}
                  >
                    <Input placeholder="Index No" />
                  </Form.Item>
                  <Form.Item
                    name="year"
                    style={{ width: '100%' }}
                    rules={[
                      {
                        required: true,
                        message: 'Please input your academic year',
                      },
                    ]}
                  >
                    <Input placeholder="Academic Year" />
                  </Form.Item>
          </div>
              <Form.Item
                  style={{ width: '100%' }}
                name="department"
                rules={[
                  { required: true, message: 'Please input your Department' },
                ]}
              >
                <Input placeholder="Department" style={{ width: '100%' }} />
              </Form.Item>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center',gap:'10px' }}>
              <Form.Item
                // label="First Name"
                  style={{ width: '100%' }}
                name="email"
                rules={[
                  {
                    required: true,
                    message: 'Please input your email address',
                  },
                ]}
              >
                <Input placeholder="email" style={{ width: '100%' }} />
              </Form.Item>
          </div>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center',gap:'10px' }}>
                  <Form.Item
                      style={{ width: '100%' }}
                    // label="First Name"
                    name="phone_num1"
                    rules={[
                      {
                        required: true,
                        message: 'Please input your phone number',
                      },
                    ]}
                  >
                    <Input placeholder="Phone number1" />
                  </Form.Item>
                  <Form.Item
                      style={{ width: '100%' }}
                    // label="First Name"
                    name="phone_num2"
                    // rules={[
                    //   { required: true, message: 'Please input your first name' },
                    // ]}
                  >
                    <Input placeholder="Phone number2" />
                  </Form.Item>
            </div>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center',gap:'10px' }}>
              <Form.Item
                  style={{ width: '100%' }}
                // label="First Name"
                name="university"
                rules={[
                  { required: true, message: 'Please input your university' },
                ]}
              >
                <Input placeholder="university" style={{ width: '100%' }} />
              </Form.Item>
          </div>

          <div style={{ marginTop: '10px' }}></div>
              <Form.Item
                // label="First Name"
                name="password_2"
                rules={[
                  { required: true, message: 'Please input your Password' },
                ]}
              >
                <Input.Password
                  placeholder="Password"
                  style={{ width: '100%' }}                />
              </Form.Item>
              <Form.Item
                // label="First Name"
                name="confirm_password_2"
                rules={[{ required: true, message: 'Please Confirm Password' }]}
              >
                <Input.Password
                  placeholder="Confirm Password"
                  style={{ width: '100%' }}                />
              </Form.Item>
          <Row>
            {isPassError && (
              <span style={{ color: 'red' }}>Password does not match</span>
            )}
          </Row>

          <Row>
            <Col lg={24}>
              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  style={{ width: '330px' }}
                >
                  Submit
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Tabs.TabPane>
      <Tabs.TabPane tab="Career Guidence Officer" key="2">
        <Form
          name="basic"
          form={form2}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          style={{ maxWidth: 600 }}
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
                    <Input placeholder="First Name" />
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
                    <Input placeholder="Last Name" />
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
                <Input placeholder="Department" style={{ width: '330px' }} />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="email"
                rules={[
                  {
                    required: true,
                    message: 'Please input your email address',
                  },
                ]}
              >
                <Input placeholder="email" style={{ width: '330px' }} />
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
                    <Input placeholder="Phone number1" />
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
                    <Input placeholder="Phone number2" />
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
                <Input placeholder="university" style={{ width: '330px' }} />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ marginTop: '10px' }}></div>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="password"
                rules={[
                  { required: true, message: 'Please input your Password' },
                ]}
              >
                <Input.Password
                  placeholder="Password"
                  style={{ width: '330px' }}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            {isPassError && (
              <span style={{ color: 'red' }}>Password does not match</span>
            )}
          </Row>
          <Row>
            <Col lg={24}>
              <Form.Item
                // label="First Name"
                name="confirm_password"
                rules={[{ required: true, message: 'Please Confirm Password' }]}
              >
                <Input.Password
                  placeholder="Confirm Password"
                  style={{ width: '330px' }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row>
            <Col lg={24}>
              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  style={{ width: '330px' }}
                >
                  Submit
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Tabs.TabPane>
    </Tabs>
  );
};

export default SignUp;
