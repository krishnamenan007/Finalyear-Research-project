
import {
  Switch,
  Route,
  Redirect,
  BrowserRouter as Router,
} from 'react-router-dom';
import Home from './pages/Home';

import Rtl from './pages/Rtl';
import Profile from './pages/Profile';
import SignUp from './pages/SignUp';
import SignIn from './pages/SignIn';
import Main from './components/layout/Main';
import 'antd/dist/antd.css';
import './assets/styles/main.css';
import './assets/styles/responsive.css';
import { useSelector } from 'react-redux';
import UserActivation from '../src/common/activationPage';
import Advertisement from './pages/advertisement';
import StudentAddPerformance from './pages/studentAddPerformance';
import StudentPerformanceTabs from './pages/studentAddPerformanceTabs';
import StudentViewPerformance from './pages/studentsViewPerformance';
import ViewEvidence from './pages/viewEvidence';
import GraphComponent from './pages/lineCharts';
import MyComponent from './pages/feedback';
function App() {
  const siginStatusCode = useSelector(
    (state) => state?.commonReducer?.siginStatusCode
  );

  const role = useSelector((state) => state?.commonReducer?.roleCode);

  return (
    <div className="App">
      <Router>
        <Switch>
          <Route path="/user/activation" exact component={UserActivation} />
          <Route path="/sign-up" exact component={SignUp} />
          <Route path="/login" exact component={SignIn} />
          <Main>
            {siginStatusCode === 200 ? (
              <>
                <Route exact path="/dashboard" component={Home} />

                {/* <Route exact path="/tables" component={Tables} /> */}
                {/* <Route exact path="/billing" component={Billing} /> */}

                {role === 'STUDENT' && (
                  <>
                    <Route
                      exact
                      path="/performance"
                      component={StudentAddPerformance}
                    />
                    <Route
                      exact
                      path="/addPerformance/gpa"
                      component={StudentPerformanceTabs}
                    />
                    <Route
                      exact
                      path="/addPerformance/selfLearning"
                      component={StudentPerformanceTabs}
                    />
                    <Route
                      exact
                      path="/addPerformance/club"
                      component={StudentPerformanceTabs}
                    />
                    <Route
                      exact
                      path="/addPerformance/sport"
                      component={StudentPerformanceTabs}
                    />

                    <Route
                      exact
                      path="/addPerformance/event"
                      component={StudentPerformanceTabs}
                    />
                  </>
                )}
                <Route
                      exact
                      path="/api/fetchData"
                      component={MyComponent}
                    />
                
                <Route
                  exact
                  path="/analyzeStudentPerformance"
                  component={StudentViewPerformance}
                />
                {role === 'CAREER_GUIDANCE_OFFICER' && (
                  <Route
                    exact
                    path="/advertisement"
                    component={Advertisement}
                  />
                )}

{role === 'CAREER_GUIDANCE_OFFICER' && (
                  <Route
                    exact
                    path="/viewevidence"
                    component={ViewEvidence}
                  />
                )}

                {role === 'STUDENT' ? (
                  <Route
                    exact
                    path="/checkPerformance"
                    component={GraphComponent}
                  />
                ) : (
                  <></>
                )}

                {/* <Route exact path="/rtl" component={Rtl} /> */}

                <Route exact path="/profile" component={Profile} />
                {role === 'STUDENT' && (
                  <Route exact path="/">
                    <Redirect to="/dashboard" />
                  </Route>
                )}
                {role === 'CAREER_GUIDANCE_OFFICER' && (
                  <Route exact path="/">
                    <Redirect to="/advertisement" />
                  </Route>
                )}
              </>
            ) : (
              <Redirect to="/login" />
            )}
          </Main>
        </Switch>
      </Router>
    </div>
  );
}

export default App;
