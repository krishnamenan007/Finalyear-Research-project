
// export default Sidenav;
// import { useState } from "react";
import { Menu, Button } from "antd";
import { NavLink, useLocation, useHistory } from "react-router-dom";
import logo from "../../assets/images/logo.png";
import { useSelector } from "react-redux";
import Swal from "sweetalert2";
import { signOut} from "../../redux/action";
import { useDispatch } from "react-redux";
import viewEvidence from "../../pages/viewEvidence";
import MyComponent from "../../pages/feedback";
function Sidenav({ color }) {
  const { pathname } = useLocation();
  const page = pathname.replace("/", "");
  const role = useSelector((state) => state?.commonReducer?.roleCode);
  const dispatch = useDispatch();

  const signOutFunc = () => {
    Swal.fire({
      title: "Are you sure you want to sign out?",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#3085d6",
      cancelButtonColor: "#d33",
      confirmButtonText: "Yes",
    }).then((result) => {
      if (result.isConfirmed) {
        if (result.isConfirmed) {
          localStorage.setItem("token", "");
          localStorage.clear();
          signOut(dispatch);
      }
    }});};

const feedback =[
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="14"  />
    <path d="M9.69627 17.6973L14.9463 22.9473L24.6963 13.1973" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <defs>
        <linearGradient
          id="paint0_linear_310_1173"
          x1="18"
          y1="0"
          x2="18"
          y2="36"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#070C83" />
          <stop offset="1" stop-color="#8DF4F4" />
        </linearGradient>
      </defs>
  </svg>];
  const dashboard = [
    <svg
      width="94"
      height="85"
      viewBox="0 0 94 85"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g filter="url(#filter0_d_306_1015)">
        <path
          d="M66.1391 34.792L52.2948 24.6693C49.5893 22.6988 45.362 22.6795 42.6777 24.65L28.8334 34.792C26.8466 36.2409 25.6418 39.1386 26.0645 41.4182L28.7277 55.9841C29.3407 59.2488 32.6591 61.8182 36.2734 61.8182H58.6779C62.25 61.8182 65.6318 59.1909 66.2448 55.9648L68.9079 41.3988C69.2884 39.1386 68.0836 36.2409 66.1391 34.792ZM49.0609 54.0909C49.0609 54.8829 48.3423 55.5398 47.4757 55.5398C46.6091 55.5398 45.8904 54.8829 45.8904 54.0909V48.2954C45.8904 47.5034 46.6091 46.8466 47.4757 46.8466C48.3423 46.8466 49.0609 47.5034 49.0609 48.2954V54.0909Z"
          fill="url(#paint0_linear_306_1015)"
        />
      </g>
      <defs>
        <filter
          id="filter0_d_306_1015"
          x="9.98022"
          y="12.1818"
          width="74.9978"
          height="70.6364"
          filterUnits="userSpaceOnUse"
          color-interpolation-filters="sRGB"
        >
          <feFlood flood-opacity="0" result="BackgroundImageFix" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha"
          />
          <feOffset dy="5" />
          <feGaussianBlur stdDeviation="8" />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0.368627 0 0 0 0 0.835294 0 0 0 0 0.658824 0 0 0 0.25 0"
          />
          <feBlend
            mode="normal"
            in2="BackgroundImageFix"
            result="effect1_dropShadow_306_1015"
          />
          <feBlend
            mode="normal"
            in="SourceGraphic"
            in2="effect1_dropShadow_306_1015"
            result="shape"
          />
        </filter>
        <linearGradient
          id="paint0_linear_306_1015"
          x1="47.4791"
          y1="23.1818"
          x2="47.4791"
          y2="61.8182"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#061888" />
          <stop offset="0.732292" stop-color="#5D8EF9" />
          <stop offset="1" stop-color="#8DF4F4" />
        </linearGradient>
      </defs>
    </svg>,
  ];

  const performance = [
    <svg
      width="40"
      height="40"
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M16.25 0C14.1789 0 12.5 1.67893 12.5 3.75V6.25C12.5 8.32107 14.1789 10 16.25 10H23.75C25.8211 10 27.5 8.32107 27.5 6.25V3.75C27.5 1.67893 25.8211 0 23.75 0H16.25ZM23.75 2.5C24.4404 2.5 25 3.05964 25 3.75V6.25C25 6.94036 24.4404 7.5 23.75 7.5H16.25C15.5596 7.5 15 6.94036 15 6.25V3.75C15 3.05964 15.5596 2.5 16.25 2.5H23.75Z"
        fill="url(#paint0_linear_310_1175)"
      />
      <path
        d="M10 3.75H7.5C4.73858 3.75 2.5 5.98857 2.5 8.75V35C2.5 37.7614 4.73858 40 7.5 40H32.5C35.2614 40 37.5 37.7614 37.5 35V8.75C37.5 5.98858 35.2614 3.75 32.5 3.75H30V6.25C30 9.70178 27.2018 12.5 23.75 12.5H16.25C12.7982 12.5 10 9.70178 10 6.25V3.75ZM27.1339 22.1339L19.6339 29.6339C19.3995 29.8683 19.0815 30 18.75 30C18.4185 30 18.1005 29.8683 17.8661 29.6339L14.1161 25.8839C13.628 25.3957 13.628 24.6043 14.1161 24.1161C14.6043 23.628 15.3957 23.628 15.8839 24.1161L18.75 26.9822L25.3661 20.3661C25.8543 19.878 26.6457 19.878 27.1339 20.3661C27.622 20.8543 27.622 21.6457 27.1339 22.1339Z"
        fill="url(#paint1_linear_310_1175)"
      />
      <defs>
        <linearGradient
          id="paint0_linear_310_1175"
          x1="20"
          y1="0"
          x2="20"
          y2="40"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#070C83" />
          <stop offset="1" stop-color="#8DF4F4" />
        </linearGradient>
        <linearGradient
          id="paint1_linear_310_1175"
          x1="20"
          y1="0"
          x2="20"
          y2="40"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#070C83" />
          <stop offset="1" stop-color="#8DF4F4" />
        </linearGradient>
      </defs>
    </svg>,
  ];
      const view = [
        <svg width="32" height="40" viewBox="0 0 32 40" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 0H4C1.79 0 0.0200005 1.79 0.0200005 4L0 36C0 38.21 1.77 40 3.98 40H28C30.21 40 32 38.21 32 36V12L20 0ZM24 32H8V28H24V32ZM24 24H8V20H24V24ZM18 14V3L29 14H18Z" fill="url(#paint0_linear_52_1046)"/>
        <defs>
        <linearGradient id="paint0_linear_52_1046" x1="16" y1="0" x2="16" y2="40" gradientUnits="userSpaceOnUse">
        <stop stop-color="#053D9A"/>
        <stop offset="1" stop-color="#6FBFDA"/>
        </linearGradient>
        </defs>
        </svg>
        
    ];

  const Analyze = [
    <svg
      width="36"
      height="36"
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        fill-rule="evenodd"
        clip-rule="evenodd"
        d="M0 0H2.25V33.75H36V36H0V0ZM22.5 7.875C22.5 7.25368 23.0037 6.75 23.625 6.75H32.625C33.2463 6.75 33.75 7.25368 33.75 7.875V16.875C33.75 17.4963 33.2463 18 32.625 18C32.0037 18 31.5 17.4963 31.5 16.875V11.0266L23.3707 20.9624C23.1694 21.2084 22.8735 21.3578 22.556 21.3736C22.2386 21.3894 21.9293 21.2702 21.7045 21.0455L15.8855 15.2264L7.65983 26.5367C7.29438 27.0392 6.59079 27.1503 6.08831 26.7848C5.58582 26.4194 5.47473 25.7158 5.84017 25.2133L14.8402 12.8383C15.0337 12.5722 15.334 12.4042 15.662 12.3784C15.99 12.3527 16.3128 12.4719 16.5455 12.7045L22.4165 18.5755L30.251 9H23.625C23.0037 9 22.5 8.49632 22.5 7.875Z"
        fill="url(#paint0_linear_310_1173)"
      />
      <defs>
        <linearGradient
          id="paint0_linear_310_1173"
          x1="18"
          y1="0"
          x2="18"
          y2="36"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#070C83" />
          <stop offset="1" stop-color="#8DF4F4" />
        </linearGradient>
      </defs>
    </svg>,
  ];


  const profile = [
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      key={0}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M18 10C18 14.4183 14.4183 18 10 18C5.58172 18 2 14.4183 2 10C2 5.58172 5.58172 2 10 2C14.4183 2 18 5.58172 18 10ZM12 7C12 8.10457 11.1046 9 10 9C8.89543 9 8 8.10457 8 7C8 5.89543 8.89543 5 10 5C11.1046 5 12 5.89543 12 7ZM9.99993 11C7.98239 11 6.24394 12.195 5.45374 13.9157C6.55403 15.192 8.18265 16 9.99998 16C11.8173 16 13.4459 15.1921 14.5462 13.9158C13.756 12.195 12.0175 11 9.99993 11Z"
        fill={color}
      ></path>
    </svg>,
  ];

  const signin = [
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      key={0}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M6 2C5.44772 2 5 2.44772 5 3V4H4C2.89543 4 2 4.89543 2 6V16C2 17.1046 2.89543 18 4 18H16C17.1046 18 18 17.1046 18 16V6C18 4.89543 17.1046 4 16 4H15V3C15 2.44772 14.5523 2 14 2C13.4477 2 13 2.44772 13 3V4H7V3C7 2.44772 6.55228 2 6 2ZM6 7C5.44772 7 5 7.44772 5 8C5 8.55228 5.44772 9 6 9H14C14.5523 9 15 8.55228 15 8C15 7.44772 14.5523 7 14 7H6Z"
        fill={color}
      ></path>
    </svg>,
  ];

  const signup = [
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="14"
      height="14"
      viewBox="0 0 14 14"
      key={0}
    >
      <path
        d="M0,2A2,2,0,0,1,2,0H8a2,2,0,0,1,2,2V8a2,2,0,0,1-2,2H2A2,2,0,0,1,0,8Z"
        transform="translate(4 4)"
        fill={color}
      />
      <path
        d="M2,0A2,2,0,0,0,0,2V8a2,2,0,0,0,2,2V4A2,2,0,0,1,4,2h6A2,2,0,0,0,8,0Z"
        fill={color}
      />
    </svg>,
  ];

  const analyzeStudents = [
    <svg
      width="35"
      height="35"
      viewBox="0 0 35 35"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M24.0625 4.375C24.0625 3.16688 25.0419 2.1875 26.25 2.1875H30.625C31.8331 2.1875 32.8125 3.16688 32.8125 4.375V30.625H33.9062C34.5103 30.625 35 31.1147 35 31.7188C35 32.3228 34.5103 32.8125 33.9062 32.8125H1.09375C0.489689 32.8125 0 32.3228 0 31.7188C0 31.1147 0.489689 30.625 1.09375 30.625H2.1875V24.0625C2.1875 22.8544 3.16688 21.875 4.375 21.875H8.75C9.95812 21.875 10.9375 22.8544 10.9375 24.0625V30.625H13.125V15.3125C13.125 14.1044 14.1044 13.125 15.3125 13.125H19.6875C20.8956 13.125 21.875 14.1044 21.875 15.3125V30.625H24.0625V4.375Z"
        fill="url(#paint0_linear_268_863)"
      />
      <defs>
        <linearGradient
          id="paint0_linear_268_863"
          x1="17.5"
          y1="2.1875"
          x2="17.5"
          y2="32.8125"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#070C83" />
          <stop offset="1" stop-color="#69D0F0" stop-opacity="0.81" />
        </linearGradient>
      </defs>
    </svg>,
  ];

  const history = useHistory();

  const handleFeedbackClick = () => {
    // Perform any necessary actions before redirecting, if needed
    // ...
    history.push('/api/fetchData');

    // Delay the goBack action by a short duration (e.g., 100 milliseconds)
     setTimeout(() => {
      // Redirect back to the previous page
       history.goBack();
     }, 1000);
  };

  const hideAboveResolution = window.innerWidth >= 768 ? { display: 'none' } : {};
  return (
    <>
      <div className="brand">
        {/* <img src={logo} alt="" />
        <span>Muse Dashboard</span> */}
      </div>
      <hr />
      <Menu theme="light" mode="inline">
        <Menu.Item key="1">
          <NavLink to="/dashboard">
            <span
              className="icon"
              style={{
                background: page === "dashboard" ? color : "",
              }}
            >
              {dashboard}
            </span>
            <span className="label">Dashboard</span>
          </NavLink>
        </Menu.Item>


       
       

        {role === "CAREER_GUIDANCE_OFFICER" ? (
          <Menu.Item key="7">
            <NavLink to="/advertisement">
              <span
                className="icon"
                style={{
                  background: page === "tables" ? color : "",
                }}
              >
                {performance}
              </span>
              <span className="label">Advertisement</span>
            </NavLink>
          </Menu.Item>
        ) : (
          <></>
        )}

        {role === "STUDENT" ? (
          <Menu.Item key="8">
            <NavLink to="/performance">
              <span
                className="icon"
                style={{
                  background: page === "tables" ? color : "",
                }}
              >
                {performance}
              </span>
              <span className="label">Performance</span>
            </NavLink>
          </Menu.Item>
        ) : (
          <></>
        )}
        
          <Menu.Item key="8">
            
              <span
                className="icon"
                style={{
                  background: page === "tables" ? color : "",
                  marginLeft:'12px'
                }}
              >
                {feedback}
              </span>
              <span className="label"
              style={{
              backgroundColor: 'darkblue', // Button background color
              color: 'white', // Text color
              fontWeight: 'bold', // Bold text
              padding: '5px 5px ', // Adjust padding as needed
              border: 'none', // Remove border
              borderRadius: '5px', // Add rounded corners
              cursor: 'pointer',
              marginTop:'2%',
            
              marginLeft:'4px',
              
              width: '150%', }}
              onClick={handleFeedbackClick}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = 'lightblue'; // Change background color on hover
                e.target.style.color = 'darkblue'; // Change text color on hover
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'darkblue'; // Change background color back on mouse leave
                e.target.style.color = 'white'; // Change text color back on mouse leave
              }}

              >Trigger Feedback</span>
          </Menu.Item>
        
          <Menu.Item key="9">
           <NavLink to="/analyzeStudentPerformance">
            <span
              className="icon"
              style={{
                background: page === "tables" ? color : "",
              }}
            >
              {analyzeStudents}
            </span>
            <span className="label">Analyze Students</span>
           </NavLink>
         </Menu.Item> 
           
         <Menu.Item className="menu-item-header" key="5">
          Account
        </Menu.Item>

        {role === "CAREER_GUIDANCE_OFFICER" ? (
                    <Menu.Item key="7" style={{margin:'10px'}}>
                        <NavLink to="/viewevidence">
              <span
                  className="icon"
                  style={{
                      background: page === "tables" ? color : "",
                      width: '64px',
                      height: '64px',
                  }}
              >
                {view}
              </span>
              <span className="label">View Evidence</span>
                        </NavLink>
                    </Menu.Item>
                ) : (
                    <></>
                )}
       

        <Menu.Item key="6">
          <NavLink to="/profile">
            <span
              className="icon"
              style={{
                background: page === "profile" ? color : "",
              }}
            >
              {profile}
            </span>
            <span className="label">Profile</span>
          </NavLink>
        </Menu.Item>

       <Menu.Item key="6" style={hideAboveResolution}>
          
            <span
            
              style={{
                background: page === "profile" ? color : "",
                display: "inline-block",
    width: "35px",
    height: "35px",
    borderRadius: "20%",
    textAlign: "center",
    lineHeight: "40px",
    fontSize: "24px",
    color: "#fff",
    marginRight: "30px",
    display:'none',
    //backgroundColor: "blue",
              }}
            >
              {profile}
            </span>
            <span className="label">
            <button onClick={signOutFunc}
            style={{
              backgroundColor: 'darkblue', // Button background color
      color: 'white', // Text color
      fontWeight: 'bold', // Bold text
      padding: '5px 10px', // Adjust padding as needed
      border: 'none', // Remove border
      borderRadius: '5px', // Add rounded corners
      cursor: 'pointer',
      marginTop:'2%',
    
      marginLeft:'15px',
      
      width: '40%', 
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'lightblue'; // Change background color on hover
              e.target.style.color = 'darkblue'; // Change text color on hover
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'darkblue'; // Change background color back on mouse leave
              e.target.style.color = 'white'; // Change text color back on mouse leave
            }}
            >Sign Out</button></span>
          
        </Menu.Item>

        
      </Menu>
      
    </>
  );
}

export default Sidenav;
