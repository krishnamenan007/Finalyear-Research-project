const { DataTypes } = require('sequelize');

module.exports = model;

function model(sequelize) {
  const attributes = {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      autoIncrement: true,
      primaryKey: true,
      //   primaryKey: true,
    },
    uuid: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    phone_num1: { type: DataTypes.INTEGER, allowNull: true },
    phone_num2: { type: DataTypes.INTEGER, allowNull: true },
    email: { type: DataTypes.TEXT, allowNull: true },
    password: { type: DataTypes.TEXT, allowNull: true },
    role_id: {
      type: DataTypes.INTEGER,
      allowNull: true,
      references: {
        model: 'roles',
        key: 'id',
      },
    },
    index_no: { type: DataTypes.INTEGER, allowNull: true },
    year: { type: DataTypes.TEXT, allowNull: true },
    status: { type: DataTypes.TEXT, allowNull: true },
    first_name: { type: DataTypes.TEXT, allowNull: true },
    last_name: { type: DataTypes.TEXT, allowNull: true },
    department: {
      type: DataTypes.TEXT,
      allowNull: true,
    },
    university: { type: DataTypes.TEXT, allowNull: true },
    image: { type: DataTypes.BLOB, allowNull: true },
  };

  return sequelize.define('users', attributes, {
    timestamps: true,
  });
}
