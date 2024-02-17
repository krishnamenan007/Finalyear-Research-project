import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Swal from 'sweetalert2';
import { useHistory } from 'react-router-dom';

import { activateUser } from '../redux/action';

const UserActivation = () => {
  const history = useHistory();
  const dispatch = useDispatch();
  const actStatus = useSelector((state) => state?.commonReducer?.actStatus);
  const activationStatus = useSelector(
    (state) => state?.commonReducer?.activationStatus
  );

  useEffect(() => {
    console.log(activationStatus, actStatus, 'testtt');
    if (activationStatus === 'now_active') {
      Swal.fire({
        title: 'User has been Verified Successfully',
        text: "You won't be able to revert this!",
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
    } else if (activationStatus === 'already_active') {
      Swal.fire({
        title: 'User has been Verified already,Please login',
        // text: "You won't be able to revert this!",
        icon: 'warning',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Login',
      }).then((result) => {
        if (result.isConfirmed) {
          history.push('/login');
        }
      });
    }
    if (actStatus === false) {
      Swal.fire({
        title: 'Error Occurred While Activating',
        // text: "You won't be able to revert this!",
        icon: 'error',
        showCancelButton: false,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Login',
      }).then((result) => {
        if (result.isConfirmed) {
          history.push('/login');
        }
      });
    }
  }, [actStatus]);
  useEffect(() => {
    const url = window.location.href;
    const uuid = url.split('=')[1];

    activateUser({ uuid }, dispatch);
  }, []);

  return <></>;
};

export default UserActivation;
