const { DataTypes } = require('sequelize');

module.exports = model;

function model(sequelize) {
  const attributes = {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      autoIncrement: true,
      primaryKey: true,
    },
    knowledge: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    problem_solving: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    communication: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    descision_making: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    creativity: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    team_work: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    leadership: {
      allowNull: true,
      type: DataTypes.FLOAT,
    },
    performance_level: {
      allowNull: true,
      type : DataTypes.TEXT,
    },
    user_id: {
      allowNull: true,
      type: DataTypes.INTEGER,
      references: {
        model: 'users',
        key: 'id',
      },
    },
  };

  return sequelize.define('skills', attributes, {
    timestamps: true,
  });
}
