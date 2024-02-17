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
    event_name: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    certificate_no:{
      type: DataTypes.TEXT,
      allowNull: true,
    },
    level: { type: DataTypes.TEXT, allowNull: true },
    cardinality: { type: DataTypes.TEXT, allowNull: true },
    position: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    file_id: {
      allowNull: true,
      type: DataTypes.INTEGER,
      allowNull: true,
      references: {
        model: 'files',
        key: 'id',
      },
    },
    user_id: {
      allowNull: false,
      type: DataTypes.INTEGER,
      allowNull: true,
      references: {
        model: 'users',
        key: 'id',
      },
    },
  };

  return sequelize.define('events', attributes, {
    timestamps: true,
  });
}
