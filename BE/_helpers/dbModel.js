const config = require("./config.json");
const mysql = require("mysql2/promise");
const { Sequelize } = require("sequelize");

module.exports = db = {};

initialize();

async function initialize() {
  // create db if it doesn't already exist
  const { host, port, user, password, database } = config.database;
  const connection = await mysql.createConnection({
    host,
    port,
    user,
    password,
  });
  await connection.query(`CREATE DATABASE IF NOT EXISTS \`${database}\`;`);

  // connect to db
  const sequelize = new Sequelize(database, user, password, {
    dialect: "mysql",
  });

  // init models and add them to the exported db object
  db.User = require("../models/user.model")(sequelize);
  db.Role = require("../models/role.model")(sequelize);
  db.Advertisement = require("../models/advertisement.model")(sequelize);
  db.SelfLearning = require("../models/selfLearning.model")(sequelize);
  db.File = require("../models/file.model")(sequelize);
  db.GPA = require("../models/gpa.model")(sequelize);
  db.Skill = require("../models/skillModel")(sequelize);
  db.Club = require("../models/clubModel")(sequelize);
  db.Event = require("../models/event.model")(sequelize);
  db.Sport = require("../models/sport.model")(sequelize);

  db.User.belongsTo(db.Role, { foreignKey: "role_id" });

  db.User.hasMany(db.Club, { foreignKey: "user_id" });
  db.User.hasMany(db.Event, { foreignKey: "user_id" });

  db.User.hasMany(db.Sport, { foreignKey: "user_id" });

  db.User.hasMany(db.SelfLearning, { foreignKey: "user_id" });
  db.User.hasMany(db.GPA, { foreignKey: "user_id" });
  db.User.hasMany(db.Skill, { foreignKey: "user_id" });
  db.Advertisement.belongsTo(db.User, { foreignKey: "created_user_Id" });
  db.GPA.belongsTo(db.File, { foreignKey: "file_id" });
  db.GPA.belongsTo(db.User, { foreignKey: "user_id" });
  db.Club.belongsTo(db.File, { foreignKey: "file_id" });
  db.Club.belongsTo(db.User, { foreignKey: "user_id" });

  db.Sport.belongsTo(db.File, { foreignKey: "file_id" });
  db.Sport.belongsTo(db.User, { foreignKey: "user_id" });
  db.Event.belongsTo(db.File, { foreignKey: "file_id" });
  db.Event.belongsTo(db.User, { foreignKey: "user_id" });
  db.SelfLearning.belongsTo(db.File, { foreignKey: "file_id" });
  db.SelfLearning.belongsTo(db.User, { foreignKey: "user_id" });
  db.Skill.belongsTo(db.User, { foreignKey: "user_id" });
  // sync all models with database
  await sequelize.sync();
}
