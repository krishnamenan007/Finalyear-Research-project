import ReactApexChart from "react-apexcharts";
import { Typography } from "antd";
import { MinusOutlined } from "@ant-design/icons";
import lineChart from "./configs/lineChart";
import React, { Component } from "react";

function LineChart() {
  const { Title, Paragraph } = Typography;

  return (
    <>
      {/*<div className="linechart">*/}
      {/*  <div>*/}
      {/*    <Title level={5}>Active Students</Title>*/}
      {/*    <Paragraph className="lastweek">*/}
      {/*      than last week <span className="bnb2">+30%</span>*/}
      {/*    </Paragraph>*/}
      {/*  </div>*/}
      {/*  <div className="sales">*/}
      {/*    <ul>*/}
      {/*      <li>{<MinusOutlined />} Student Performnace</li>*/}
      {/*      /!* <li>{<MinusOutlined />} Sales</li> *!/*/}
      {/*    </ul>*/}
      {/*  </div>*/}
      {/*</div>*/}

      <ReactApexChart
        options={lineChart.options}
        series={lineChart.series}
        type="bar"
        height={350}
        width={"100%"}
      />
    </>
  );
}

export default LineChart;
