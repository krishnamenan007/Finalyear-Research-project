import actionTypes from '../actionType';

const initialState = {
  getStudentStatus: null,
  getStudentData: [],
  profileStatus: null,
  profileData: null,
  updateProfilePicStatus: null,
  updateProfileInfoStatus: null,
  deleteStuStatus: null,
  updateStuStatus: null,
};

const studentReducer = (state = initialState, action) => {
  switch (action.type) {
    case actionTypes.UPDATE_PROFILE_START:
      return {
        ...state,
        updateProfilePicStatus: null,
      };

    case actionTypes.UPDATE_PROFILE_SUCCESS:
      return {
        ...state,
        updateProfilePicStatus: true,
      };

    case actionTypes.UPDATE_PROFILE_FAIL:
      return {
        ...state,
        updateProfilePicStatus: false,
      };

    case actionTypes.GET_STUDENTS_START:
      return {
        ...state,
        getStudentStatus: null,
        getStudentData: [],
      };

    case actionTypes.GET_STUDENTS_SUCCESS:
      return {
        ...state,
        getStudentStatus: true,
        getStudentData: action.payload,
      };

    case actionTypes.GET_STUDENTS_FAIL:
      return {
        ...state,
        getStudentStatus: false,
        getStudentData: [],
      };

    case actionTypes.DELETE_AD_SUCCESS:
      return {
        ...state,
        deleteAdStatus: true,
      };

    case actionTypes.GET_PROFILE_START:
      return {
        ...state,
        profileStatus: null,
      };

    case actionTypes.GET_PROFILE_FAIL:
      return {
        ...state,
        profileStatus: false,
        profileData: false,
      };

    case actionTypes.GET_PROFILE_SUCCESS:
      return {
        ...state,
        profileStatus: true,
        profileData: action.payload,
      };

    case actionTypes.UPDATE_INFO_PROFILE_START:
      return {
        ...state,
        updateProfileInfoStatus: null,
      };

    case actionTypes.UPDATE_INFO_PROFILE_SUCCESS:
      return {
        ...state,
        updateProfileInfoStatus: true,
      };

    case actionTypes.UPDATE_INFO_PROFILE_FAIL:
      return {
        ...state,
        updateProfileInfoStatus: false,
      };

    case actionTypes.UPDATE_STU_START:
      return {
        ...state,
        updateStuStatus: null,
      };

    case actionTypes.UPDATE_STU_SUCCESS:
      return {
        ...state,
        updateStuStatus: true,
      };

    case actionTypes.UPDATE_STU_FAIL:
      return {
        ...state,
        updateStuStatus: false,
      };

    case actionTypes.DELETE_STU_START:
      return {
        ...state,
        deleteStuStatus: null,
      };

    case actionTypes.DELETE_STU_SUCCESS:
      return {
        ...state,
        deleteStuStatus: true,
      };

    case actionTypes.DELETE_STU_FAIL:
      return {
        ...state,
        deleteStuStatus: false,
      };
    default:
      return state;
  }
};

export default studentReducer;
