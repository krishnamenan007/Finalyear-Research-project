import actionTypes from '../actionType';

const initialState = {
  gpaStatus: null,
};

const studentDetailReducer = (state = initialState, action) => {
  switch (action.type) {
    case actionTypes.GPA_START:
      return {
        ...state,
        gpaStatus: null,
      };
    case actionTypes.GPA_SUCCESS:
      return {
        ...state,
        gpaStatus: true,
      };
    case actionTypes.GPA_FAIL:
      return {
        ...state,
        gpaStatus: false,
      };

    default:
      return state;
  }
};

export default studentDetailReducer;
