const sql = require("../_helpers/db.js");
const tableName = "connections";

// constructor
const connections = function (connection) {
  this.branchID = connection.branchID;
  this.areaCode = connection.areaCode;
  this.roadID = connection.roadID;
  this.connectionAddress = connection.connectionAddress;
  this.connectionID = connection.connectionID;
  this.oldID = connection.oldID;
  this.connectionLocation = connection.connectionLocation;
  this.status = connection.status;
  this.connectedDate = connection.connectedDate;
  this.connectionStatus = connection.connectionStatus;
  this.connectionType = connection.connectionType;
  this.tvCount = connection.tvCount;
  this.remarks = connection.remarks;
  this.actionDate = connection.actionDate;
  this.customerID = connection.customerID;
  this.dueAmount = connection.dueAmount;
};

const ModelName = connections;

ModelName.create = (newConnections, result) => {
  sql.query(`INSERT INTO ${tableName} SET ?`, newConnections, (err, res) => {
    if (err) {
      result(err, null);
      return;
    }

    result(null, { id: res.insertId, ...newConnections });
    var getID = res.insertId;
    var finalID = ("00000" + getID).slice(-5);
    const areaid = newConnections.areaCode;

    sql.query(`SELECT * FROM areas WHERE id= '${areaid}'`, (err, res) => {
      if (err) {
        result(err, null);
        return;
      }
      const areacode = res[0].areaCode;
      sql.query(
        `UPDATE ${tableName} SET connectionID = ? WHERE id = ${getID} `,
        [areacode + finalID]
      );
    });
  });
};

ModelName.findbyId = (id, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and cn.id = '${id}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbycustomerId = (customerId, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and c.id = '${customerId}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbyconnectionId = (connectionId, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and connectionID = '${connectionId}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbyOldId = (oldId, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and oldID = '${oldId}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbyNIC = (nic, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and subscriberNIC = '${nic}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbyPhone = (phone, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and primaryPhone = '${phone}'`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findbyName = (name, result) => {
  sql.query(
    `SELECT * FROM customers c,connections cn WHERE c.id = cn.customerID and (firstName = '${name}' or  lastName = '${name}')`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.getAllByAdminID = (adminId, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE adminId = ${adminId}`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res[0]);
        return;
      }

      // not found connectionss with the id

      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.getAllByJobNameAdminId = (adminId, jobnameId, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE adminId = ${adminId} && jobName = ${jobnameId}`,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res[0]);
        return;
      }

      // not found connectionss with the id
      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.getAll = (result) => {
  sql.query(`SELECT * FROM ${tableName}`, (err, res) => {
    if (err) {
      result(null, err);
      return;
    }

    result(null, res);
  });
};
ModelName.findAllSubscriber = (result) => {
  sql.query(`SELECT * FROM customers c,connections cn  WHERE c.id = cn.customerID`, 
  (err, res) => {
    if (err) {
      result(null, err);
      return;
    }

    result(null, res);
  });
};
ModelName.updateById = (id, connections, result) => {
  this.branchID = connections.branchID;
  this.areaCode = connections.areaCode;
  this.roadID = connections.roadID;
  this.connectionID = connections.connectionID;
  // this.phone = connections.phone;
  this.connectionAddress = connections.connectionAddress;
  this.oldID = connections.oldID;
  this.connectionLocation = connections.connectionLocation;
  this.status = connections.status;
  this.connectedDate = connections.connectedDate;
  this.connectionStatus = connections.connectionStatus;

  this.connectionType = connections.connectionType;
  this.tvCount = connections.tvCount;
  this.remarks = connections.remarks;
  this.actionDate = connections.actionDate;
  this.dueAmount = connections.dueAmount;

  sql.query(
    `UPDATE ${tableName} SET branchID = ?
    , areaCode = ? 
    , roadID = ? 
    , connectionID = ? 
    , connectionAddress = ? 
    , oldID = ? 
    , connectionLocation = ? 
    , status = ? 

    , connectedDate = ? 
    , connectionStatus = ? 
    , connectionType = ?
    , tvCount = ?
    ,remarks = ?
    ,actionDate = ?
    ,dueAmount = ?


    WHERE connectionID = '${id}'`,
    [
      connections.branchID,
      connections.areaCode,
      connections.roadID,
      connections.connectionID,
      // connections.phone,
      connections.connectionAddress,
      connections.oldID,
      connections.connectionLocation,
      connections.status,
      connections.connectedDate,
      connections.connectionStatus,
      connections.connectionType,
      connections.tvCount,
      connections.remarks,
      connections.actionDate,
      connections.dueAmount,

      id,
    ],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }

      result(null, { id: id, ...connections });
    }
  );
};


// update location
ModelName.updateLocation = (id, connections, result) => {
  this.connectionLocation = connections.connectionLocation;
  sql.query(
    `UPDATE ${tableName} SET  connectionLocation = ? 
    WHERE connectionID = '${id}'`,
    [connections.connectionLocation, id],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }
      result(null, { id: id, ...connections });
    }
  );
};

//update status Active
ModelName.updateStatusActive = (id, connections,connectionID,enteredBy,conductedBy, result) => {
  this.connectionStatus = connections.connectionStatus;
  this.actionDate = connections.actionDate;
  this.dueAmount = connections.dueAmount;
  this.remarks = connections.remarks;
  sql.query(
    `UPDATE ${tableName} SET 
     connectionStatus = ? 
    , actionDate = ? 
    , remarks = ? 
    WHERE connectionID = '${id}'`,
    [
      "Active",
      connections.actionDate,
      "Reconnection",
      id,
    ],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }

      result(null, { id: id, ...connections });

      sql.query(`SELECT * FROM ${tableName} WHERE connectionID= '${id}'`, (err, res) => {
      if (err) {
        result(err, null);
        return;
      }
        const due = res[0].dueAmount;
      
        sql.query(
        `INSERT INTO payments (connectionID, paidDateTime, paymentType, amount, due, enteredBy, conductedBy) VALUES ('${connectionID}','${connections.actionDate}', 'RECONNECTION', '${connections.dueAmount}', '${due}', '${enteredBy}', '${conductedBy}' )`
        , (err, res) => {
    if (err) {
      result(err, null);
      return;
    }
          console.log(res);
          console.log(res.insertId);
          console.log(res.insertId);
          

    const getdeactivationpaymentID = res.insertId;
   
          const finaldeactivationpaymentID = ("00000000" + getdeactivationpaymentID).slice(-8);
          console.log(finaldeactivationpaymentID);
    sql.query(
      `UPDATE payments SET invoiceID = ? WHERE id = ${res.insertId} `,
      ["INV" + finaldeactivationpaymentID]
    );
  }
      );

      });
      
      

    }
  );
};

//update status Deactivate
ModelName.updateStatusDeactivate = (id, connections,connectionID,enteredBy,conductedBy, result) => {
  this.connectionStatus = connections.connectionStatus;
  this.actionDate = connections.actionDate;
  this.remarks = connections.remarks;
  sql.query(
    `UPDATE ${tableName} SET 
     connectionStatus = ? 
    , actionDate = ? 
    , remarks = ? 
    WHERE connectionID = '${id}'`,
    [
      "Deactivate",
      connections.actionDate,
      connections.remarks,
      id,
    ],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }

      result(null, { id: id, ...connections });

      sql.query(`SELECT * FROM ${tableName} WHERE connectionID= '${id}'`, (err, res) => {
      if (err) {
        result(err, null);
        return;
      }
        const due = res[0].dueAmount;
      
        sql.query(
        `INSERT INTO payments (connectionID, paidDateTime, paymentType, amount, due, enteredBy, conductedBy) VALUES ('${connectionID}','${connections.actionDate}', 'DEACTIVATED', '0', '${due}', '${enteredBy}', '${conductedBy}' )`
        , (err, res) => {
    if (err) {
      result(err, null);
      return;
    }
          console.log(res);
          console.log(res.insertId);
          console.log(res.insertId);
          

    const getdeactivationpaymentID = res.insertId;
   
          const finaldeactivationpaymentID = ("00000000" + getdeactivationpaymentID).slice(-8);
          console.log(finaldeactivationpaymentID);
    sql.query(
      `UPDATE payments SET invoiceID = ? WHERE id = ${res.insertId} `,
      ["INV" + finaldeactivationpaymentID]
    );
  }
      );

      });

      

    }
  );
};

//update TV count
ModelName.updateTVCount = (id, connections, result) => {
  this.tvCount = connections.tvCount;
  this.remarks = connections.remarks;
  sql.query(
    `UPDATE ${tableName} SET 
     remarks = ? 
    WHERE connectionID = '${id}'`,
    [
      "Extra TV",
      id,
    ],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }

      result(null, { id: id, ...connections });
      console.log(connections);
      sql.query(`SELECT * FROM ${tableName} WHERE connectionID= '${id}'`, (err, res) => {
      if (err) {
        result(err, null);
        return;
      }
      const tv = res[0].tvCount;
      sql.query(
        `UPDATE ${tableName} SET tvCount = ? WHERE connectionID = '${id}' `,
        [tv + connections.tvCount]
      );
    });

    }
  );
};

// change Address
ModelName.changeAddress = (id, connections, connectionID,enteredBy,conductedBy, result) => {
  this.areaCode = connections.areaCode;
  this.roadID = connections.roadID;
  this.connectionAddress = connections.connectionAddress;
  this.connectionID = connections.connectionID;
  this.connectionLocation = connections.connectionLocation;
  this.remarks = connections.remarks;
  this.dueAmount = connections.dueAmount;
  this.actionDate = connections.actionDate;
  sql.query(
    `UPDATE ${tableName} SET  areaCode = ? 
    , roadID = ? 
    , connectionAddress = ? 
    , connectionLocation = ? 
    , remarks = ? 
    , actionDate = ? 
    WHERE id = '${id}'`,
    [
      connections.areaCode,
      connections.roadID,
      connections.connectionAddress,
      null,
      "Change Location",
      connections.actionDate,
      id],
    (err, res) => {
      if (err) {
        result(null, err);
        return;
      }

      if (res.affectedRows == 0) {
        // not found connections with the id
        result({ kind: "not_found" }, null);
        return;
      }
      result(null, { id: id, ...connections });

      sql.query(`SELECT * FROM ${tableName} WHERE id= '${id}'`, (err, res) => {
      if (err) {
        result(err, null);
        return;
      }
        
        var getID = res[0].connectionID.slice(-5);
        console.log(getID);
        const areaid = connections.areaCode;
        var due = res[0].dueAmount;

        sql.query(`SELECT * FROM areas WHERE id= '${areaid}'`, (err, res) => {
          if (err) {
            result(err, null);
            return;
          }
          const areacode = res[0].areaCode;
          sql.query(
            `UPDATE ${tableName} SET connectionID = ?
             WHERE id = '${id}'`,
            [
              areacode + getID,
            ]
          );
        });

       sql.query(
        `INSERT INTO payments (connectionID, paidDateTime, paymentType, amount, due, enteredBy, conductedBy) VALUES ('${connectionID}','${connections.actionDate}', 'CHANGE LOCATION', '${connections.dueAmount}', '${due}', '${enteredBy}', '${conductedBy}' )`
        , (err, res) => {
    if (err) {
      result(err, null);
      return;
    }
          console.log(res);
          console.log(res.insertId);
          console.log(res.insertId);
          

    const getdeactivationpaymentID = res.insertId;
   
          const finaldeactivationpaymentID = ("00000000" + getdeactivationpaymentID).slice(-8);
          console.log(finaldeactivationpaymentID);
    sql.query(
      `UPDATE payments SET invoiceID = ? WHERE id = ${res.insertId} `,
      ["INV" + finaldeactivationpaymentID]
    );
  }
      );

      });
      
       

    }
  );
};



ModelName.remove = (id, result) => {
  sql.query(`DELETE FROM ${tableName} WHERE id = ?`, id, (err, res) => {
    if (err) {
      result(null, err);
      return;
    }

    if (res.affectedRows == 0) {
      // not found connections with the id
      result({ kind: "not_found" }, null);
      return;
    }

    result(null, res);
  });
};

// Get Due reports by Area , road, status and minimum due
ModelName.findAllFilterReport = (area, road, status, dueAmount, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && roadID = ${road} && connectionStatus = '${status}' && dueAmount >= ${dueAmount} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area
ModelName.findAreaFilterReport = (area, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area and road,
ModelName.findAreaandRoadFilterReport = (area, road, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && roadID = ${road}  `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area and road,
ModelName.findAreaRoadandStatusFilterReport = (area, road, status, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && roadID = ${road} && connectionStatus = '${status}' `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

ModelName.findAreaandStatusFilterReport = (area, status, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area}  && connectionStatus = '${status}' `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area and road,
ModelName.findAreaRoadandDueFilterReport = (area, road, status, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && roadID = ${road} && dueAmount >= ${dueAmount} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area and Status,
ModelName.findAreaandStatusFilterReport = (area, connectionStatus, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && connectionStatus = '${connectionStatus}'  `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area, Status and Due
ModelName.findAreaStatusandDueFilterReport = (area, connectionStatus, dueAmount, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && connectionStatus = '${connectionStatus}' && dueAmount >= ${dueAmount}  `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area
ModelName.findAreaandDueFilterReport = (area, dueAmount, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && dueAmount >= ${dueAmount} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports by Area , road, status and minimum due
ModelName.findAlFilterReport = (area, road, status, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE areaCode = ${area} && roadID = ${road} && status = ${status} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};




// Get Due reports bystatus
ModelName.findAllFilterStatusReport = (status, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE status = '${status}' `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};

// Get Due reports amount
ModelName.findAllFilterMinAmountReport = (amount, result) => {
  sql.query(
    `SELECT * FROM ${tableName} WHERE dueAmount > ${amount} `,
    (err, res) => {
      if (err) {
        result(err, null);
        return;
      }

      if (res.length) {
        result(null, res);
        return;
      }

      // not found connections with the id

      result({ kind: "not_found" }, null);
    }
  );
};


// Get unpaid reports by area
ModelName.findAllUnpaidConnection = (result) => {
  sql.query(`SELECT * FROM connections  WHERE dueAmount > 500 `, (err, res) => {
    if (err) {
      result(err, null);
      return;
    }

    if (res.length) {
      result(null, res);
      return;
    }

    // not found connections with the id

    result({ kind: "not_found" }, null);
  });
};
module.exports = ModelName;
