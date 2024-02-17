import actionTypes from '../actionType';

const initialState = {
  adCreateStatus: null,
  getAdStatus: null,
  adData: [],
  updateAdStatus: null,
  deleteAdStatus: null,
  adByDateStatus: null,
  adByDateData: [],
};

const advertisementReducer = (state = initialState, action) => {
  switch (action.type) {
    case actionTypes.AD_CREATE_START:
      return {
        ...state,
        adCreateStatus: null,
      };

    case actionTypes.AD_CREATE_SUCCESS:
      return {
        ...state,
        adCreateStatus: true,
      };

    case actionTypes.AD_CREATE_FAIL:
      return {
        ...state,
        adCreateStatus: false,
      };

    case actionTypes.GET_AD_START:
      return {
        ...state,
        getAdStatus: null,
      };

    case actionTypes.GET_AD_SUCCESS:
      return {
        ...state,
        getAdStatus: true,
        adData: action.payload,
      };

    case actionTypes.GET_AD_FAIL:
      return {
        ...state,
        getAdStatus: false,
      };

    case actionTypes.UPDATE_AD_START:
      return {
        ...state,
        updateAdStatus: null,
      };

    case actionTypes.UPDATE_AD_FAIL:
      return {
        ...state,
        updateAdStatus: false,
      };

    case actionTypes.UPDATE_AD_SUCCESS:
      return {
        ...state,
        updateAdStatus: true,
      };

    case actionTypes.DELETE_AD_START:
      return {
        ...state,
        deleteAdStatus: null,
      };

    case actionTypes.DELETE_AD_FAIL:
      return {
        ...state,
        deleteAdStatus: false,
      };

    case actionTypes.DELETE_AD_SUCCESS:
      return {
        ...state,
        deleteAdStatus: true,
      };

    case actionTypes.GET_AD_BY_DATE_START:
      return {
        ...state,
        adByDateStatus: null,
        adByDateData: [],
      };

    case actionTypes.GET_AD_BY_DATE_FAIL:
      return {
        ...state,
        adByDateStatus: false,
        adByDateData: [],
      };

    case actionTypes.GET_AD_BY_DATE_SUCCESS:
      return {
        ...state,
        adByDateStatus: true,
        adByDateData: action.payload,
      };
    default:
      return state;
  }
};

export default advertisementReducer;
