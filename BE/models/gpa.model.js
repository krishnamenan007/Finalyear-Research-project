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
    value: {
      type: DataTypes.DOUBLE,
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
      allowNull: true,
      type: DataTypes.INTEGER,
      allowNull: true,
      references: {
        model: 'users',
        key: 'id',
      },
    },
  };

  return sequelize.define('gpas', attributes, {
    timestamps: true,
  });
}
