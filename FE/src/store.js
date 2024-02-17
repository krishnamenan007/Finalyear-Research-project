import { createStore, applyMiddleware } from 'redux';
import rootReducer from '../src/redux/reducers/rootReducer';
import thunk from 'redux-thunk';
import { compose } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension';
import storage from 'redux-persist/lib/storage';
import { persistStore, persistReducer } from 'redux-persist';

const persistConfig = {
  key: 'reducer',
  storage: storage,
 
};
const presistedReducer = persistReducer(persistConfig, rootReducer);
const store = createStore(
  presistedReducer,
  composeWithDevTools(applyMiddleware(thunk))
);
const persistor = persistStore(store);
export { persistor, store };
