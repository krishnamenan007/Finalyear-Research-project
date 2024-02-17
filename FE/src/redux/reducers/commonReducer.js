import actionTypes from '../actionType';

const initialState = {
  loading: false,
  isSignup: null,
  toastrStatus: null,
  toastrMSg: {
    message: '',
    statusCode: null,
  },
  activationStatus: '',
  actStatus: null,
  siginStatusCode: null,
  signinSatus: null,
  resendStatus: false,
  roleCode: '',
  signupMessage: '',
};

const commonReducer = (state = initialState, action) => {
  switch (action.type) {
    case actionTypes.RESEND_START:
      return {
        ...state,
        resendStatus: null,
      };

    case actionTypes.RESEND_SUCCESS:
      return {
        ...state,
        resendStatus: true,
      };

    case actionTypes.RESEND_FAILURE:
      return {
        ...state,
        resendStatus: false,
      };

    case actionTypes.SIGNIN_START:
      return {
        ...state,
        signinSatus: null,
        siginStatusCode: null,
      };

    case actionTypes.SIGNIN_SUCCESS:
      return {
        ...state,
        signinSatus: true,
        siginStatusCode: action.payload.statusCode,
        roleCode: action.payload.roleCode,
      };

    case actionTypes.SIGNIN_ERROR:
      return {
        ...state,
        signinSatus: false,
        siginStatusCode: action.payload,
      };
    case actionTypes.LOADING_START:
      return {
        ...state,
        loading: true,
      };
    case actionTypes.LOADING_END:
      return {
        ...state,
        loading: false,
      };

    case actionTypes.SIGN_UP_START:
      return {
        ...state,
        isSignup: null,
        signupMessage: '',
      };

    case actionTypes.SIGN_UP_SUCCESS:
      return {
        ...state,
        isSignup: true,
        signupMessage: '',
      };

    case actionTypes.SIGN_UP_FAILURE:
      return {
        ...state,
        isSignup: false,
        signupMessage: action.payload,
      };

    case actionTypes.TOASTR_SUCCESS:
      return {
        ...state,
        toastrStatus: true,
        toastrMSg: action.payload,
      };

    case actionTypes.TOASTR_START:
      return {
        ...state,
        toastrStatus: null,
        toastrMSg: {},
      };

    case actionTypes.TOASTR_FAILURE:
      return {
        ...state,
        toastrStatus: false,
        toastrMSg: {
          statusCode: 500,
          message: action.payload,
        },
      };

    case actionTypes.ACTIVATION_START:
      return {
        ...state,
        actStatus: null,
        activationStatus: action.payload,
      };
    case actionTypes.ACTIVATION_SUCCESS:
      return {
        ...state,
        actStatus: true,
        activationStatus: action.payload,
      };
    case actionTypes.ACTIVATION_FAILURE:
      return {
        ...state,
        actStatus: false,
        activationStatus: action.payload,
      };
    default:
      return state;
  }
};

export default commonReducer;
