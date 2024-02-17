import commonReducer from './commonReducer';
import advertisementReducer from './advertisementReducer';

import { combineReducers } from 'redux';
import studentDetailReducer from './studentDetailReducer';
import studentReducer from './studentReducer';

const rootReducer = combineReducers({
  commonReducer: commonReducer,
  advertisementReducer: advertisementReducer,
  studentDetailReducer: studentDetailReducer,
  studentReducer: studentReducer,
});

export default rootReducer;
