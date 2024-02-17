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
    role_code: { type: DataTypes.TEXT, allowNull: false },
    role_name: { type: DataTypes.TEXT, allowNull: false },
  };

  return sequelize.define('roles', attributes, {
    timestamps: false,
  });
}
