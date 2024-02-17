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
    file_name: { type: DataTypes.TEXT, allowNull: false },
    file_data: { type: DataTypes.BLOB, allowNull: false },
  };

  return sequelize.define('files', attributes, {
    timestamps: true,
  });
}
