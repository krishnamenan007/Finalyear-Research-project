// import React from 'react';
// import { Layout, Menu } from 'antd';
// import { Link, useLocation } from 'react-router-dom';
// import {
//   DesktopOutlined,
//   PieChartOutlined,
//   UserOutlined,
//   TeamOutlined,
//   FileOutlined,
// } from '@ant-design/icons';
// import DashboardIcon from './studentsDetail/Icons/DashboardIcon';
// import PerformanceIcon from './studentsDetail/Icons/PerformanceIcon';
// import GraphIcon from './studentsDetail/Icons/GraphIcon';
// // import { Sidebar, Menu, MenuItem } from 'react-pro-sidebar';

// const { Sider } = Layout;

// const AppSidebar = () => {
//   const location = useLocation();

//   const getMenuKeyFromPathname = (pathname) => {
//     if (pathname.startsWith('/dashboard')) return '1';
//     if (pathname.startsWith('/users')) return '2';
//     if (pathname.startsWith('/teams')) return '3';
//     if (pathname.startsWith('/devices')) return '4';
//     if (pathname.startsWith('/files')) return '5';
//     return null;
//   };

//   const selectedKey = getMenuKeyFromPathname(location.pathname);

//   return (
//     <Sider
//       collapsible
//       width={120} // Set the width as per your requirement
//       style={{
//         backgroundColor: '#E6E7FF',
//       }}
//     >
//       <Menu
//         selectedKeys={selectedKey ? [selectedKey] : []}
//         mode="inline"
//         style={{ backgroundColor: '#E6E7FF', marginTop: '20px' }}
//       >
//         <table>
//           <tbody>
//             <tr>
//               <td>
//                 <div style={{ width: '30px' }}></div>
//               </td>
//               <td>
//                 <Menu.Item
//                   key="1"
//                   icon={<DashboardIcon />}
//                   style={{ marginBottom: '10px' }}
//                 >
//                   <Link to="/dashboard"></Link>
//                 </Menu.Item>
//               </td>
//             </tr>
//             <tr>
//               <td></td>
//             </tr>
//           </tbody>
//         </table>

//         <table>
//           <tbody>
//             <tr>
//               <td>
//                 <div style={{ width: '60px' }}></div>
//               </td>
//               <td>
//                 <Menu.Item
//                   key="2"
//                   icon={<PerformanceIcon />}
//                   style={{ marginBottom: '10px' }}
//                 >
//                   <Link to="/advertisement"></Link>
//                 </Menu.Item>
//               </td>
//             </tr>
//           </tbody>
//         </table>

//         <table>
//           <tbody>
//             <tr>
//               <td>
//                 <div style={{ width: '60px' }}></div>
//               </td>
//               <td>
//                 <Menu.Item
//                   key="3"
//                   icon={<GraphIcon />}
//                   style={{ marginBottom: '10px' }}
//                 >
//                   <Link to="/addPerformance"></Link>
//                 </Menu.Item>
//               </td>
//             </tr>
//           </tbody>
//         </table>
//         <table>
//           <tbody>
//             <tr>
//               <td>
//                 <div style={{ width: '60px' }}></div>
//               </td>
//               <td>
//                 <Menu.Item
//                   key="3"
//                   icon={<GraphIcon />}
//                   style={{ marginBottom: '10px' }}
//                 >
//                   <Link to="/profile"></Link>
//                 </Menu.Item>
//               </td>
//             </tr>
//           </tbody>
//         </table>
//       </Menu>
//     </Sider>
//   );
// };

// export default AppSidebar;
