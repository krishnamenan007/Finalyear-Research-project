const { DataTypes } = require("sequelize");

module.exports = model;

function model(sequelize) {
  const attributes = {
    branchID: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: {
        model: "branches",
        key: "id",
      },
    },
    areaCode: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: {
        model: "areas",
        key: "id",
      },
    },
    roadID: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: {
        model: "roads",
        key: "id",
      },
    },
    customerID: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: {
        model: "customers",
        key: "id",
      },
    },
    connectionAddress: { type: DataTypes.STRING, allowNull: true },
    connectionID: { type: DataTypes.STRING, allowNull: true },
    oldID: { type: DataTypes.STRING, allowNull: false },
    connectionLocation: { type: DataTypes.STRING, allowNull: true },
    status: { type: DataTypes.STRING, allowNull: true },
    connectedDate: { type: DataTypes.STRING, allowNull: true },
    connectionStatus: { type: DataTypes.STRING, allowNull: true },
    connectionType: { type: DataTypes.STRING, allowNull: true },
    tvCount: { type: DataTypes.INTEGER, allowNull: true },
    remarks: { type: DataTypes.STRING, allowNull: true },
    dueAmount: { type: DataTypes.INTEGER, allowNull: true },
    actionDate: { type: DataTypes.DATE, allowNull: true },
  };

  return sequelize.define("connection", attributes, {
    timestamps: false,
  });
}
