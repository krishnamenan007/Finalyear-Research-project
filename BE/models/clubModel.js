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
    club_name: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    certificate_no:{
      type: DataTypes.TEXT,
      allowNull: true,
    },
    club_level: { type: DataTypes.TEXT, allowNull: true },
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

  return sequelize.define('clubs', attributes, {
    timestamps: true,
  });
}
