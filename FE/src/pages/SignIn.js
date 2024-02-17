import React, {Component, useEffect, useState} from 'react';
import {Link} from 'react-router-dom';
import {
    Layout,
    Menu,
    Button,
    Row,
    Col,
    Typography,
    Form,
    Input,
    Switch,
} from 'antd';
import signinbg from '../assets/images/graduation_image.png';
import logopng from '../assets/images/logo.png';
import {
    DribbbleOutlined,
    TwitterOutlined,
    InstagramOutlined,
    GithubOutlined,
} from '@ant-design/icons';
import {useHistory} from 'react-router-dom/cjs/react-router-dom';
import {useDispatch, useSelector} from 'react-redux';
import {makeStatusDefault, resendLink, siginIn} from '../redux/action';
import Swal from 'sweetalert2';

function onChange(checked) {
    console.log(`switch to ${checked}`);
}

const {Title} = Typography;
const {Header, Footer, Content} = Layout;

const SignIn = () => {
    const history = useHistory();
    const [input, setInput] = useState({});
    const dispatch = useDispatch();
    const siginStatusCode = useSelector(
        (state) => state?.commonReducer?.siginStatusCode
    );
    const resendStatus = useSelector(
        (state) => state?.commonReducer?.resendStatus
    );

    useEffect(() => {
        if (resendStatus === true) {
            Swal.fire({
                title: 'we have send you the email with activation link please verify',
                icon: 'success',
                showCancelButton: false,
                confirmButtonText: 'OK',
            }).then((result) => {
                if (result.isConfirmed) {
                    makeStatusDefault(dispatch);
                }
                makeStatusDefault(dispatch);
            });
        }
    }, [resendStatus]);

    useEffect(() => {
        console.log(siginStatusCode, 'siginStatusCode');
        if (siginStatusCode === 401) {
            Swal.fire({
                title: 'user not found ',
                icon: 'error',
                showCancelButton: false,
            });
            makeStatusDefault(dispatch);
        } else if (siginStatusCode === 402) {
            Swal.fire({
                title: 'User is not active ',
                // text: "You won't be able to revert this!",
                icon: 'error',
                showCancelButton: false,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Send Activation Email',
            }).then((result) => {
                if (result.isConfirmed) {
                    resendLink({email: input?.email}, dispatch);
                }
            });
            makeStatusDefault(dispatch);
        } else if (siginStatusCode === 403) {
            Swal.fire({
                title: 'Password is incorrect',

                icon: 'error',
                showCancelButton: true,
            });
            makeStatusDefault(dispatch);
        } else if (siginStatusCode === 200) {
            history.push('/');
        }
    }, [siginStatusCode]);

    const onFinish = (values) => {
        setInput(values);
        siginIn(values, dispatch);
    };

    return (
        <>
            <Layout className="layout-default layout-signin full-width" style={{minHeight: '100vh'}}>
                {/*Logo on Left Top*/}
                <div className="logo" style={{position: 'absolute', top: '5vh', left: '20vh', padding: '20px'}}>
                    <img src={logopng} alt="logo" style={{width: '60px',marginBottom: '20px'}}/>
                    <div className="logo__text">
                        <h4 style={{color: '#070C83'}}>Hey There,</h4>
                        <h1 style={{color: '#070C83', fontWeight: 'bolder'}}>
                            Welcome Back
                        </h1>
                        <h6 style={{color: '#070C83', width: '80%'}}>
                            We design  your path way to successful internships
                        </h6>
                    </div>
                </div>

                <Content className="signin" style={{height: '100vh'}}>
                    <Row
                        gutter={[24, 0]}
                        justify="end"
                        style={{width: '90%', margin: '0',height: 'fit-content',display: 'flex',alignItems: 'center'}}
                    >
                        <Col
                            style={{background: '#fff', padding: '50px 50px', width: '30%',height: '90%',borderRadius: '30px'}}
                        >
                            {/* Sign In and Sign Up Selection Bar */}
                            <Row
                                gutter={[24, 0]}
                                justify="center"
                                style={{width: '100%', margin: '0',background: '#E6E7FF', padding: '10px',borderRadius: '30px'}}
                            >
                                <Col span={24} style={{textAlign: 'center', display: 'flex', justifyContent: 'center', gap: '10px',fontSize: '20px'}}>
                                        <Link to="/sign-up" className="signin-signup__item" style={{width: '50%',background: '#fff',borderRadius: '30px',padding: '5px 0px',color: '#000AF3'}}>
                                            Sign Up
                                        </Link>
                                        <Link to="/login" className="signin-signup__item active" style={{width: '50%',background: '#000AF3',borderRadius: '30px',padding: '5px 0px',color: '#fff'}}>
                                            Sign In
                                        </Link>
                                </Col>
                            </Row>

                            <Title className="mb-15 " style={{textAlign: 'center', marginTop: '20px'}} level={3}>
                                Sign In
                            </Title>
                            <Form
                                onFinish={onFinish}
                                // onFinishFailed={onFinishFailed}
                                layout="vertical"
                                className="row-col"
                            >
                                <Form.Item
                                    className="username"
                                    label="Email"
                                    name="email"
                                    rules={[
                                        {
                                            required: true,
                                            message: 'Please input your email!',
                                        },
                                    ]}
                                >
                                    <Input placeholder="Email"/>
                                </Form.Item>

                                <Form.Item
                                    className="username"
                                    label="Password"
                                    name="password"
                                    rules={[
                                        {
                                            required: true,
                                            message: 'Please input your password!',
                                        },
                                    ]}
                                >
                                    <Input.Password placeholder="Password"/>
                                </Form.Item>

                                <Form.Item
                                    name="remember"
                                    className="aligin-center"
                                    valuePropName="checked"
                                >
                                    <Switch defaultChecked onChange={onChange}/>
                                    Remember me
                                </Form.Item>

                                <Form.Item>
                                    <Button
                                        type="primary"
                                        htmlType="submit"
                                        style={{width: '100%'}}
                                    >
                                        SIGN IN
                                    </Button>
                                </Form.Item>
                                <p className="font-semibold text-muted">
                                    Don't have an account?{' '}
                                    <Link to="/sign-up" className="text-dark font-bold">
                                        Sign Up
                                    </Link>
                                </p>
                            </Form>
                        </Col>
                    </Row>
                </Content>
            </Layout>
        </>
    );
};

export default SignIn;
