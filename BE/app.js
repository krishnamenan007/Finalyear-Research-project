require("rootpath")();
const express = require("express");
const app = express();
const verifyToken = require("./_middleware/authorize");
const cors = require("cors");
const bodyParser = require("body-parser");
const errorHandler = require("_middleware/error-handler");
const multer = require("multer");
const path = require("path");
const fetchDataMiddleware = require('./controllers/conn');
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(cors());

// LANDIGN PAGE
app.get("/", function (req, res) {
  res.sendFile(__dirname + "/views/index.html");
});

// API ROUTES
app.use("/auth", require("./controllers/signupController"));
app.use("/advertisements", require("./controllers/advertisementController"));
app.use("/gpa", require("./controllers/skillsmappingController"));
app.use("/profile", require("./controllers/profileController"));
app.use("/students", require("./controllers/studentsController"));
app.use("/approvel", require("./controllers/approvelController"));
app.use('/api/fetchData', fetchDataMiddleware);
// 404 PAGE - always should be last
app.get("*", function (req, res) {
  res.status(404).sendFile(__dirname + "/views/notfound.html");
});

// global error handler
app.use(errorHandler);

// start server
const port =
  process.env.NODE_ENV === "production" ? process.env.PORT || 80 : 4000;
app.listen(port, () => console.log("Server listening on port " + port));
