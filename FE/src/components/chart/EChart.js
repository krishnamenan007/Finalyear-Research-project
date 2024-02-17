/*!
  =========================================================
  * Muse Ant Design Dashboard - v1.0.0
  =========================================================
  * Product Page: https://www.creative-tim.com/product/muse-ant-design-dashboard
  * Copyright 2021 Creative Tim (https://www.creative-tim.com)
  * Licensed under MIT (https://github.com/creativetimofficial/muse-ant-design-dashboard/blob/main/LICENSE.md)
  * Coded by Creative Tim
  =========================================================
  * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

import ReactApexChart from 'react-apexcharts';
import { Row, Col, Typography } from 'antd';
import eChart from './configs/eChart';

function EChart() {
  const { Title, Paragraph } = Typography;

  const items = [
    {
      Title: '3,6K',
      user: 'Users',
    },
    {
      Title: '2m',
      user: 'Clicks',
    },
    {
      Title: '$772',
      user: 'Sales',
    },
    {
      Title: '82',
      user: 'Items',
    },
  ];

  return (
    <>
      <div
        style={{ color: '#070C83', fontWeight: 600, fontSize: '30px' }}
        id="chart"
      >
        Come Join with us!
      </div>
      <div
        className="chart-vistior"
        style={{ color: '#070C83', fontWeight: 600, fontSize: '15px' }}
      >
        {/* <Title level={5}>Active Users</Title> */}
        <Paragraph
          className="lastweek"
          style={{ color: '#070C83', fontWeight: 600, fontSize: '15px' }}
        >
          It may take some time to achieve your goals, but eventually, you will.
          So be patient and work hard for your dreams, and they will come to
          you.
        </Paragraph>
      </div>
    </>
  );
}

export default EChart;
