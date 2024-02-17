var app = require("express").Router();
const connection = require("./connection.controller.js");
const subUrl = "connection";
const subUrl1 = "connectionReport";

// Create a new connection
app.post(`/${subUrl}`, connection.create);

// Retrieve all connection
app.get(`/${subUrl}`, connection.findAll);

// // Retrieve all connection & subscriber
app.get(`/${subUrl}/subscriber`, connection.findAllSubscriber);

// Retrieve a single connection with id
app.get(`/${subUrl}/id/:id`, connection.findOnebyId);

// Retrieve a single connection with connectionId
app.get(`/${subUrl}/customerId/:customerId`, connection.findOnebycustomerId);

// Retrieve a single connection with connectionId
app.get(`/${subUrl}/connectionId/:connectionId`, connection.findOnebyconnectionId);

// Retrieve a single connection with oldId
app.get(`/${subUrl}/oldId/:oldId`, connection.findOnebyOldId);

// Retrieve a single connection with NIC
app.get(`/${subUrl}/nic/:nic`, connection.findOnebyNIC);

// Retrieve a single connection with phone
app.get(`/${subUrl}/phone/:phone`, connection.findOnebyPhone);

// Retrieve a single connection with name
app.get(`/${subUrl}/name/:name`, connection.findOnebyName);

// Update a connection with connectionId
app.put(`/${subUrl}/:connectionId`, connection.update);

//update connection location
app.put(`/${subUrl}/location/:connectionId`, connection.updateCustomerLocation);

//update connectionStatus Active
app.put(`/${subUrl}/status/active/:connectionId`, connection.updateConnectionStatusActive);

//update connectionStatus Deactivate
app.put(`/${subUrl}/status/deactivate/:connectionId`, connection.updateConnectionStatusDeactivate);

//update TV Count
app.put(`/${subUrl}/tvcount/:connectionId`, connection.updateConnectionTVCount);

// Update a connection with connectionId
app.put(`/${subUrl}/address/:id`, connection.changeConnectionAddress);

// Delete a connection with connectionId
app.delete(`/${subUrl}/:connectionId`, connection.delete);

app.get(`/${subUrl1}/area/:area/`, connection.findAreaFilterReport);

app.get(`/${subUrl1}/areaandroad/:area/:road/`, connection.findAreaandRoadFilterReport);

app.get(`/${subUrl1}/arearoadandstatus/:area/:road/:status/`, connection.findAreaRoadandStatusFilterReport);

app.get(`/${subUrl1}/arearoadanddue/:area/:road/:dueAmount`, connection.findAreaRoadandDueFilterReport);

app.get(`/${subUrl1}/areaandstatus/:area/:connectionStatus`, connection.findAreaandStatusFilterReport);

app.get(`/${subUrl1}/areastatusanddue/:area/:connectionStatus/:dueAmount`, connection.findAreaStatusandDueFilterReport);

app.get(`/${subUrl1}/areaanddue/:area/:dueAmount`, connection.findAreaandDueFilterReport);

app.get(
  `/${subUrl1}/:area/:road/:status/:dueAmount`,
  connection.findAllFilterReport
);



app.get(
  `/${subUrl}/reports/status/:status/`,
  connection.findAllFilterStatusReport
);

app.get(
  `/${subUrl}/reports/min-amount/:amount/`,
  connection.findAllFilterMinAmountReport
);
// amount;
app.get(`/${subUrl}/reports/unpaid`, connection.findAllUnpaidConnection);

module.exports = app;
