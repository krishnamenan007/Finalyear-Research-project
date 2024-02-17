import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { persistor, store } from './store';
import Loading from '../src/common/loadingAnimation';

ReactDOM.render(
  <>
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        {/* <Toastr /> */}
        <Loading />
        <App />
      </PersistGate>
    </Provider>
  </>,
  document.getElementById('root')
);
