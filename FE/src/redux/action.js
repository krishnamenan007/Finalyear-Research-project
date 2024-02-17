import axios from 'axios';
import actionTypes from './actionType';

export const siginIn = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });
  dispatch({
    type: actionTypes.SIGNIN_START,
  });
  const config = {
    url: 'http://localhost:4000/auth/signin',
    method: 'POST',
    data: payload,
  };
  try {
    const resu = await axios(config);
    if ([401, 402, 403].includes(resu.data?.statusCode)) {
      dispatch({
        type: actionTypes.SIGNIN_ERROR,
        payload: resu.data?.statusCode,
      });
    } else if (resu.data?.statusCode === 200) {
      dispatch({
        type: actionTypes.SIGNIN_SUCCESS,
        payload: { statusCode: 200, roleCode: resu?.data?.data?.role_code },
      });
      localStorage.setItem('token', resu?.data?.data?.token);
    }
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });
    dispatch({
      type: actionTypes.SIGNIN_ERROR,
      payload: 500,
    });
  }
};

export const signUp = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });
  dispatch({
    type: actionTypes.SIGN_UP_START,
  });

  dispatch({
    type: actionTypes.TOASTR_START,
  });

  const config = {
    url: 'http://localhost:4000/auth/signup',
    method: 'POST',
    data: payload,
    headers: { 'content-type': 'multipart/form-data' },
  };
  try {
    const result = await axios(config);
    console.log(result, 'res');
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.SIGN_UP_SUCCESS,
    });
  } catch (error) {
    console.log(error);
    dispatch({
      type: actionTypes.SIGN_UP_FAILURE,
      payload: error?.response?.data?.message,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  }
};

export const defaultSignUp = (dispatch) => {
  dispatch({
    type: actionTypes.SIGN_UP_START,
  });
};

export const activateUser = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });
  dispatch({
    type: actionTypes.ACTIVATION_START,
    payload: '',
  });
  const config = {
    url: 'http://localhost:4000/auth/activate',
    method: 'POST',
    data: payload,
  };

  try {
    const result = await axios(config);
    console.log(result?.data?.statusCode, 'lll');
    console.log(result, 'res');
    dispatch({
      type: actionTypes.LOADING_END,
    });
    dispatch({
      type: actionTypes.ACTIVATION_SUCCESS,
      payload:
        result?.data?.statusCode === 200 ? 'now_active' : 'already_active',
    });
  } catch (error) {
    dispatch({
      type: actionTypes.ACTIVATION_FAILURE,
      payload: '',
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  }
};

export const resendLink = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.SIGNIN_START,
  });
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.RESEND_START,
  });

  const config = {
    url: 'http://localhost:4000/auth/resendLink',
    method: 'post',
    data: payload,
  };

  try {
    await axios(config);
    dispatch({
      type: actionTypes.LOADING_END,
    });
    dispatch({
      type: actionTypes.RESEND_SUCCESS,
    });
  } catch (err) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.RESEND_FAILURE,
    });
  }
};

export const signOut = (dispatch) => {
  dispatch({
    type: actionTypes.SIGNIN_ERROR,
    payload: 500,
  });
};

export const makeStatusDefault = (dispatch) => {
  dispatch({
    type: actionTypes.SIGNIN_START,
  });
  dispatch({
    type: actionTypes.ACTIVATION_START,
  });
  dispatch({
    type: actionTypes.RESEND_START,
  });
  dispatch({
    type: actionTypes.SIGN_UP_START,
  });
};

export const createAD = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.AD_CREATE_START,
  });
  const config = {
    url: 'http://localhost:4000/advertisements',
    method: 'POST',
    data: payload,
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);
    dispatch({
      type: actionTypes.AD_CREATE_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getAllAds(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.AD_CREATE_FAIL,
    });
  }
};

export const getAllAds = async (dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GET_AD_START,
  });
  const config = {
    url: 'http://localhost:4000/advertisements',
    method: 'GET',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    const res = await axios(config);
    dispatch({
      type: actionTypes.GET_AD_SUCCESS,
      payload: res?.data?.data || [],
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.GET_AD_FAIL,
    });
  }
};

export const updateAd = async (id, payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.UPDATE_AD_START,
  });
  const config = {
    url: `http://localhost:4000/advertisements/${id}`,
    data: payload,
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);
    dispatch({
      type: actionTypes.UPDATE_AD_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getAllAds(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.UPDATE_AD_FAIL,
    });
  }
};

export const deleteAd = async (id, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.DELETE_AD_START,
  });
  const config = {
    url: `http://localhost:4000/advertisements/${id}`,
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);
    dispatch({
      type: actionTypes.DELETE_AD_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getAllAds(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.DELETE_AD_FAIL,
    });
  }
};

export const initailizeAd = (dispatch) => {
  dispatch({
    type: actionTypes.UPDATE_AD_START,
  });
  dispatch({
    type: actionTypes.GET_AD_START,
  });
  dispatch({
    type: actionTypes.AD_CREATE_START,
  });
  dispatch({
    type: actionTypes.DELETE_AD_START,
  });
};

export const submitGPAData = async (payload, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GPA_START,
  });
  const config = {
    url: `http://localhost:4000/gpa`,
    data: payload,
    method: 'POST',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
      'content-type': 'multipart/form-data',
    },
  };

  try {
    await axios(config);
    dispatch({
      type: actionTypes.GPA_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getProfileDetails(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.GPA_FAIL,
    });
  }
};

export const intializeStudDetails = (dispatch) => {
  dispatch({
    type: actionTypes.GPA_START,
  });
};

export const getAllStudents = async (query, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GET_STUDENTS_START,
  });
  const token =localStorage.getItem('token');
  const config = {
    url: `http://localhost:4000/students?token=`+token,
    params: query,
    method: 'GET',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
      'content-type': 'aaplication/json',
    },
  };

  try {
    const results = await axios(config);

    console.log(results, 'stu>>>>');
    dispatch({
      type: actionTypes.GET_STUDENTS_SUCCESS,
      payload: results?.data,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.GET_STUDENTS_FAIL,
    });
  }
};


// ------------------ New ----------------------------

export const getApprovelDetails = async (query, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GET_PROFILE_DETAILS_START,
  });

  const token = localStorage.getItem('token');
  const config = {
    url: `http://localhost:4000/approvel`, // Updated URL to /approvel
    params: query,
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  };

  try {
    const results = await axios(config);
    console.log('API Response:', results)

    dispatch({
      type: actionTypes.GET_PROFILE_DETAILS_SUCCESS,
      payload: results?.data,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    console.error('API Error', error);
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.GET_PROFILE_DETAILS_FAIL,
    });
  }
};



export const getProfileDetails = async (dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GET_PROFILE_START,
  });
  const config = {
    url: `http://localhost:4000/profile`,
    method: 'GET',
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    const results = await axios(config);

    dispatch({
      type: actionTypes.GET_PROFILE_SUCCESS,
      payload: results?.data,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.GET_PROFILE_FAIL,
    });
  }
};

export const updateProfilePicDetails = async (data, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.UPDATE_PROFILE_START,
  });
  const config = {
    url: `http://localhost:4000/profile/updatePic`,
    method: 'PUT',
    data,
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
      'content-type': 'multipart/form-data',
    },
  };

  try {
    await axios(config);

    dispatch({
      type: actionTypes.UPDATE_PROFILE_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getProfileDetails(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.UPDATE_PROFILE_FAIL,
    });
  }
};

export const updateProfileDetails = async (data, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.UPDATE_INFO_PROFILE_START,
  });
  const config = {
    url: `http://localhost:4000/profile/updateInfo`,
    method: 'PUT',
    data,
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);

    dispatch({
      type: actionTypes.UPDATE_INFO_PROFILE_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getProfileDetails(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.UPDATE_INFO_PROFILE_FAIL,
    });
  }
};

export const makeDeafultProfiles = (dispatch) => {
  dispatch({
    type: actionTypes.UPDATE_PROFILE_START,
  });
  dispatch({
    type: actionTypes.UPDATE_INFO_PROFILE_START,
  });

  dispatch({
    type: actionTypes.UPDATE_STU_START,
  });
  dispatch({
    type: actionTypes.DELETE_STU_START,
  });
};

export const updateStu = async (data, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.UPDATE_STU_START,
  });
  const config = {
    url: `http://localhost:4000/profile/updateStudentDetails`,
    method: 'PUT',
    data,
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);

    dispatch({
      type: actionTypes.UPDATE_STU_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getProfileDetails(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.UPDATE_STU_FAIL,
    });
  }
};

export const deleteStu = async (data, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.DELETE_STU_START,
  });
  const config = {
    url: `http://localhost:4000/profile/deleteStuInfo`,
    method: 'PUT',
    data,
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    await axios(config);

    dispatch({
      type: actionTypes.DELETE_STU_SUCCESS,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
    getProfileDetails(dispatch);
  } catch (error) {
    dispatch({
      type: actionTypes.LOADING_END,
    });

    dispatch({
      type: actionTypes.DELETE_STU_FAIL,
    });
  }
};

export const getAdsByDate = async (data, dispatch) => {
  dispatch({
    type: actionTypes.LOADING_START,
  });

  dispatch({
    type: actionTypes.GET_AD_BY_DATE_START,
  });
  const config = {
    url: `http://localhost:4000/advertisements/getAdsByDate`,
    method: 'POST',
    data: {
      date: data,
    },
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
  };

  try {
    const results = await axios(config);

    dispatch({
      type: actionTypes.GET_AD_BY_DATE_SUCCESS,
      payload: results.data,
    });
    dispatch({
      type: actionTypes.LOADING_END,
    });
  } catch (error) {
    dispatch({
      type: actionTypes.GET_AD_BY_DATE_FAIL,
    });

    dispatch({
      type: actionTypes.LOADING_END,
    });
  }
};
