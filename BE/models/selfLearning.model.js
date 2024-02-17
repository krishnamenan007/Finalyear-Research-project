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
    course_name: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    course_level: { type: DataTypes.TEXT, allowNull: true },
    file_id: {
      allowNull: true,
      type: DataTypes.INTEGER,
      references: {
        model: 'files',
        key: 'id',
      },
    },
    certificate_no:{
      type: DataTypes.TEXT,
      allowNull: true,
    },
    user_id: {
      allowNull: true,
      type: DataTypes.INTEGER,
      references: {
        model: 'users',
        key: 'id',
      },
    },
    approved: {
      type: DataTypes.BOOLEAN,
      allowNull: false,
      defaultValue: false,
    },
    rejected: {
      type: DataTypes.BOOLEAN,
      allowNull: false,
      defaultValue: false,
    },
  };

  return sequelize.define('selflearning', attributes, {
    timestamps: true,
  });
}
