const mysql = require('mysql2');
const dbConfignew = require('./config.json');

// Create a connection to the database
const connection = mysql.createConnection({
  host: dbConfignew.database.host,
  user: dbConfignew.database.user,
  password: dbConfignew.database.password,
  database: dbConfignew.database.database,
  multipleStatements: true,
});
// 843e7ed91b2e14957180808deeb7c60d1f4203156b118671
module.exports = connection;
