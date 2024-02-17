import React, { Component } from 'react';
import {
  Layout,
  Menu,
  Button,
  Typography,
  Card,
  Form,
  Input,
  Checkbox, Row, Col,
} from 'antd';

import { Link } from 'react-router-dom';

import SignUp from './signupComponent';
import logopng from "../assets/images/logo.png";

const { Title } = Typography;
const { Header, Footer, Content } = Layout;
const SignupComponent = () => {
  return (
    <>
      <Layout className="layout-default layout-signin full-width" style={{minHeight: '100vh',paddingTop:"20px"}}>
        {/*Logo on Left Top*/}
        <div className="logo" style={{marginTop:"20px",position: 'absolute', top: '5vh', left: '20vh', padding: '20px'}}>
          <img src={logopng} alt="logo" style={{width: '60px',marginBottom: '20px'}}/>
          <div className="logo__text">s
            <h4 style={{color: '#070C83'}}>Hey There,</h4>
            <h1 style={{color: '#070C83', fontWeight: 'bolder'}}>
              Let’s complete your profile !
            </h1>
            <h6 style={{color: '#070C83', width: '80%'}}>
              We design  your path way to successful internships
            </h6>
          </div>
        </div>

        <Content className="signin" style={{marginTop:"200px",height: '100vh'}}>
          <Row
              gutter={[24, 0]}
              justify="end"
              style={{width: '95%', margin: '0',marginTop: '10px',marginBottom: '10px',height: 'fit-content',display: 'flex',alignItems: 'center'}}
          >
            <Col
                style={{background: '#fff', padding: '20px 20px', width: '30%',height: '95%',borderRadius: '30px'}}
            >
              {/* Sign In and Sign Up Selection Bar */}
              <Row
                  gutter={[24, 0]}
                  justify="center"
                  style={{width: '100%', margin: '0',background: '#E6E7FF', padding: '10px',borderRadius: '30px'}}
              >
                <Col span={24} style={{textAlign: 'center', display: 'flex', justifyContent: 'center', gap: '10px',fontSize: '20px'}}>
                  <Link to="/sign-up" className="signin-signup__item" style={{width: '50%',background: '#000AF3',borderRadius: '30px',padding: '5px 0px',color: '#fff'}}>
                    Sign Up
                  </Link>
                  <Link to="/login" className="signin-signup__item active" style={{width: '50%',background: '#fff',borderRadius: '30px',padding: '5px 0px',color: '#000AF3'}}>
                    Sign In
                  </Link>
                </Col>
              </Row>

          <div
            title={<h5>Register</h5>}
          >
            <div style={{ marginLeft: '30px' }}>
              <SignUp className="mt-15" />
            </div>
            <p className="font-semibold text-muted text-center">
              Already have an account?{' '}
              <Link to="/login" className="font-bold text-dark">
                Sign In
              </Link>
            </p>
          </div>
            </Col>
            </Row>

          
        </Content>
      </Layout>

    </>
  );
};

export default SignupComponent;
