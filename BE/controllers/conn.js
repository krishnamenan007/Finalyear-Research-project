
const db = require('../_helpers/db');
const axios = require('axios');
const baseUrl = 'http://127.0.0.1:5000/classify_grade';
const express = require('express');
const app = express();
const verifyToken = require('../_middleware/authorize');
//const authorize= require('..//_middleware/riderauthorize');
const cors = require('cors');
// const router = express.Router();
// router.get("/", verifyToken, getProfileDetails);
// app.use(cors());

// async function getProfileDetails(req, res, next) {
//   const userID = req.user.sub.userId;
//    // Call fetchDataFromTable and pass userID as a parameter
//    await fetchDataFromTable(req, res, next,userID);
  
//   }


// Custom middleware function to access req.userId
// async function fetchDataFromTable(req, res, next) {

//   db.query('SELECT * FROM skills', (error, results) => {
//     if (error) {
//       res.status(500).json({ error: 'Database error' });
//     } else {
//       if (results.length > 0) {
//         // console.log('User ID from getProfileDetails:', userID);
//         //const user = req.user.sub.userId;
//         // Now you can use userID in this file as needed
// //console.log(user); // Example: Display the userID
//         // Assuming 'results' is an array of objects with fetched results
//         const row = results[0]; // Assign the first row to 'row'

//         const KW = parseFloat(row.knowledge);
//         const PS = parseFloat(row.problem_solving);
//         const CO = parseFloat(row.communication);
//         const DM = parseFloat(row.descision_making);
//         const CR = parseFloat(row.creativity);
//         const TW = parseFloat(row.team_work);
//         const LS = parseFloat(row.leadership);

//         // Define your parameters
//         const params = {
//           KW: KW,
//           LS: LS,
//           CO: CO,
//           PS: PS,
//           DM: DM,
//           CR: CR,
//           TW: TW,
//         };

//         axios
//           .get(baseUrl, { params })
//           .then((response) => {
//             console.log('Response:', response.data);

//             // Update 'row.performance_level' with the response data
//             row.performance_level = response.data;

//             db.query(
//               'UPDATE skills SET performance_level = ? WHERE id = ?',
//               [row.performance_level, row.id],
//               (error, updateResults) => {
//                 if (error) {
//                   console.error('Error updating database:', error);
//                   res.status(500).json({ error: 'Error updating database' });
//                 } else {
//                   console.log('Database updated successfully.');
//                   // Send the data to the frontend as a JSON response
                  
//           // Send the entire array of fetched data as JSON
//           res.json(results); 
//                   next(); // Call next() to move to the next middleware or route
//                 }
//               }
//             );
//           })
//           .catch((error) => {
//             console.error('Error:', error);
//             res.status(500).json({ error: 'Error making API request' });
//           });
//       } else {
//         console.log('No data found in the table.');
//         res.status(404).json({ error: 'No data found in the table' });
//       }
//     }
//   });


  
// }
async function fetchDataFromTable(req, res, next) {
  db.query('SELECT * FROM skills', async (error, results) => {
    if (error) {
      res.status(500).json({ error: 'Database error' });
    } else {
      if (results.length > 0) {
        for (let i = 0; i < results.length; i++) {
          const row = results[i]; // Get the current row
          const KW = parseFloat(row.knowledge);
          const PS = parseFloat(row.problem_solving);
          const CO = parseFloat(row.communication);
          const DM = parseFloat(row.descision_making);
          const CR = parseFloat(row.creativity);
          const TW = parseFloat(row.team_work);
          const LS = parseFloat(row.leadership);

          // Define your parameters
          const params = {
            KW: KW,
            LS: LS,
            CO: CO,
            PS: PS,
            DM: DM,
            CR: CR,
            TW: TW,
          };

          try {
            const response = await axios.get(baseUrl, { params });
            console.log('Response:', response.data);

            // Update 'row.performance_level' with the response data
            row.performance_level = response.data;

            await new Promise((resolve, reject) => {
              db.query(
                'UPDATE skills SET performance_level = ? WHERE id = ?',
                [row.performance_level, row.id],
                (error, updateResults) => {
                  if (error) {
                    console.error('Error updating database:', error);
                    reject(error);
                  } else {
                    console.log('Database updated successfully.');
                    resolve();
                  }
                }
              );
            });
          } catch (error) {
            console.error('Error:', error);
            res.status(500).json({ error: 'Error making API request or updating database' });
            return; // Exit the function early if there's an error
          }
        }

        // Send the entire array of fetched data as JSON
        res.json(results);
        next(); // Call next() to move to the next middleware or route
      } else {
        console.log('No data found in the table.');
        res.status(404).json({ error: 'No data found in the table' });
      }
    }
  });
}

module.exports = fetchDataFromTable;
