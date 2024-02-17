
import React, { useState, useEffect } from 'react';

function MyComponent() {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Make a GET request to your backend API
    fetch('http://localhost:4000/api/fetchData')
      .then((response) => response.json())
      .then((data) => {
        setData(data); // Update the state with the fetched data
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div>
      <h1>Data from Backend</h1>
      <ul>
        {data.map((item) => (
         <li key={item.id}>
         User ID: {item.user_id ?? 'Not Available'}
         <br />
         Performance Level: {item.performance_level ?? 'Not Available'}
       </li>
        ))}
      </ul>
    </div>
  );
}

export default MyComponent;
