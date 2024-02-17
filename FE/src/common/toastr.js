import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import toastr from 'toastr';
import 'toastr/build/toastr.min.css';

const Toastr = () => {
  const toastrMSg = useSelector((state) => state.commonReducer.toastrMSg);
  const toastrStatus = useSelector((state) => state.commonReducer.toastrStatus);

  console.log(toastrStatus);

  useEffect(() => {
    if (toastrStatus) {
      if (toastrMSg?.statusCode === 200) {
        toastr.success(toastrMSg?.message);
      } else {
        toastr.error(toastrMSg?.message);
      }
    } else if (toastrStatus === false) {
      if (toastrMSg?.statusCode === 500) {
        toastr.error(toastrMSg?.message);
      }
    }
  }, [toastrStatus]);

  return <></>;
};

export default Toastr;
