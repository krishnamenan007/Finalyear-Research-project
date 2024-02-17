const User = require("./connection.model.js");

const ModelName = User;

// Create and Save a new connection
exports.create = (req, res) => {
  // Validate request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }

  // Create
  const connectionClass = new User({
    branchID: req.body.branchID,
    areaCode: req.body.areaCode,
    roadID: req.body.roadID,
    customerID: req.body.customerID,
    // subscriberNIC: req.body.subscriberNIC,
    connectionAddress: req.body.connectionAddress,
    connectionID: req.body.connectionID,
    // phone: req.body.phone,
    oldID: req.body.oldID,
    connectionLocation: req.body.connectionLocation,
    status: req.body.status,
    connectedDate: req.body.connectedDate,
    connectionStatus: "Active",
    connectionType: req.body.connectionType,
    tvCount: req.body.tvCount,
    remarks: req.body.remarks,
    actionDate: req.body.actionDate,
    dueAmount: req.body.dueAmount,
  });

  // Save in the database
  ModelName.create(connectionClass, (err, data) => {
    if (err)
      res.status(500).send({
        message:
          err.message || "Some error occurred while creating the connection.",
      });
    else res.send(data);
  });
};

// Retrieve all connections from the database.
exports.findAll = (req, res) => {
  ModelName.getAll((err, data) => {
    if (err)
      res.status(500).send({
        message: err.message || "Some error occurred while retrieving connection.",
      });
    else res.send(data);
  });
};
// Retrieve all connections from the database.
exports.findAllSubscriber = (req, res) => {
  ModelName.findAllSubscriber((err, data) => {
    if (err)
      res.status(500).send({
        message: err.message || "Some error occurred while retrieving connection.",
      });
    else res.send(data);
  });
};
// Find a single connection with a id
exports.findOnebyId = (req, res) => {
  ModelName.findbyId(req.params.id, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.id,
        });
      }
    } else res.send(data);
  });
};


// Find a single connection with a customerId
exports.findOnebycustomerId = (req, res) => {
  ModelName.findbycustomerId(req.params.customerId, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with customerId " + req.params.customerId,
        });
      }
    } else res.send(data);
  });
};



// Find a single connection with a connectionId
exports.findOnebyconnectionId = (req, res) => {
  ModelName.findbyconnectionId(req.params.connectionId, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.connectionId,
        });
      }
    } else res.send(data);
  });
};

// Find a single connection with a connectionId
exports.findOnebyOldId = (req, res) => {
  ModelName.findbyOldId(req.params.oldID, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.oldID,
        });
      }
    } else res.send(data);
  });
};

// Find a single connection with a nic
exports.findOnebyNIC = (req, res) => {
  ModelName.findbyNIC(req.params.nic, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.nic,
        });
      }
    } else res.send(data);
  });
};

// Find a single connection with a phone
exports.findOnebyPhone = (req, res) => {
  ModelName.findbyPhone(req.params.phone, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.phone,
        });
      }
    } else res.send(data);
  });
};

// Find a single connection with a phone
exports.findOnebyName = (req, res) => {
  ModelName.findbyName(req.params.name, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send();
      } else {
        res.status(500).send({
          message: "Error retrieving connection with id " + req.params.name,
        });
      }
    } else res.send(data);
  });
};

// Update a connection identified by the connectionId in the request
exports.update = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }
  var imageValue;

  ModelName.updateById(
    req.params.connectionId,
    new ModelName({
      branchID: req.body.branchID,
      areaCode: req.body.areaCode,
      roadID: req.body.roadID,
      customerID: req.body.customerID,
      // subscriberNIC: req.body.subscriberNIC,
      connectionAddress: req.body.connectionAddress,
      connectionID: req.body.connectionID,
      // phone: req.body.phone,
      oldID: req.body.oldID,
      connectionLocation: req.body.connectionLocation,
      status: req.body.status,
      connectedDate: req.body.connectedDate,
      connectionStatus: req.body.connectionStatus,
      connectionType: req.body.connectionType,
      tvCount: req.body.tvCount,
      remarks: req.body.remarks,
      actionDate: req.body.actionDate,
      dueAmount: req.body.dueAmount,
    }),
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.connectionId}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.connectionId,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};



exports.updateCustomerLocation = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }

  ModelName.updateLocation(
    req.params.connectionId,
    new ModelName({
      connectionLocation: req.body.connectionLocation,
    }),
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.connectionId}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.connectionId,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};

// update connectionStatus Active
exports.updateConnectionStatusActive = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }

  ModelName.updateStatusActive(
    req.params.connectionId,
    new ModelName({
      connectionStatus: req.body.connectionStatus,
      actionDate: req.body.actionDate,
      dueAmount: req.body.dueAmount,
      remarks: req.body.remarks,
    }),
    req.body.connectionID ,
    req.body.enteredBy ,
    req.body.conductedBy ,
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.connectionId}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.connectionId,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};


// update connectionStatus Deactivate
exports.updateConnectionStatusDeactivate = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }

  ModelName.updateStatusDeactivate(
    req.params.connectionId,
    new ModelName({
      connectionStatus: req.body.connectionStatus,
      actionDate: req.body.actionDate,
      remarks: req.body.remarks,
    }),
    req.body.connectionID ,
    req.body.enteredBy ,
    req.body.conductedBy ,
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.connectionId}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.connectionId,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};


// update TV Count
exports.updateConnectionTVCount = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }

  ModelName.updateTVCount(
    req.params.connectionId,
    new ModelName({
      tvCount: req.body.tvCount,
      remarks: req.body.remarks,
    }),
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.connectionId}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.connectionId,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};


// Change Address
exports.changeConnectionAddress = (req, res) => {
  // Validate Request
  if (!req.body) {
    res.status(400).send({
      message: "Content can not be empty!",
    });
  }
  ModelName.changeAddress(
    req.params.id,
    new ModelName({
      areaCode: req.body.areaCode,
      roadID: req.body.roadID,
      connectionAddress: req.body.connectionAddress,
      connectionID: req.body.connectionID,
      connectionLocation: req.body.connectionLocation,
      remarks: req.body.remarks,
      dueAmount: req.body.dueAmount,
      actionDate: req.body.actionDate,
    }),
    req.body.connectionID ,
    req.body.enteredBy ,
    req.body.conductedBy ,
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with id ${req.params.id}.`,
          });
        } else {
          res.status(500).send({
            message: "Error updating connection with id " + req.params.id,
          });
        }
      } else res.send({ message: true, data });
    }
  );
};


// Delete a connection with the specified connectionId in the request
exports.delete = (req, res) => {
  ModelName.remove(req.params.connectionId, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with id ${req.params.connectionId}.`,
        });
      } else {
        res.status(500).send({
          message: "Could not delete connection with id " + req.params.connectionId,
        });
      }
    } else res.send({ message: `branches was deleted successfully!` });
  });
};

// Get Due reports by Area , road, status and minimum due
exports.findAllFilterReport = (req, res) => {

  ModelName.findAllFilterReport(
    req.params.area,
    req.params.road,
    req.params.status,
    req.params.dueAmount,
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with area id ${req.params.area}.`,
          });
        } else {
          res.status(500).send({
            message:
              "Error retrieving connection with area id " + req.params.area,
          });
        }
      } else res.send(data);
    }
  );
};

// Get Due reports by Area
exports.findAreaFilterReport = (req, res) => {
  ModelName.findAreaFilterReport(req.params.area, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area and road,
exports.findAreaandRoadFilterReport = (req, res) => {
  ModelName.findAreaandRoadFilterReport(
    req.params.area,
    req.params.road, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area, status and road
exports.findAreaRoadandStatusFilterReport = (req, res) => {
  ModelName.findAreaRoadandStatusFilterReport(
    req.params.area,
    req.params.road,
    req.params.status,
    (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};
// Get Due reports by Area, status 
exports.findAreaandStatusFilterReport = (req, res) => {
  ModelName.findAreaandStatusFilterReport(
    req.params.area,
    req.params.status,
    (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};
// Get Due reports by Area, status and road
exports.findAreaRoadandDueFilterReport = (req, res) => {
  ModelName.findAreaRoadandDueFilterReport(
    req.params.area,
    req.params.road,
    req.params.dueAmount,
    (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area and road,
exports.findAreaandStatusFilterReport = (req, res) => {
  ModelName.findAreaandStatusFilterReport(
    req.params.area,
    req.params.connectionStatus, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area and road,
exports.findAreaStatusandDueFilterReport = (req, res) => {
  ModelName.findAreaStatusandDueFilterReport(
    req.params.area,
    req.params.connectionStatus,
    req.params.dueAmount,
     (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area and Due
exports.findAreaandDueFilterReport = (req, res) => {
  ModelName.findAreaandDueFilterReport(req.params.area,req.params.dueAmount, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving filter connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by Area , road, status and minimum due
exports.findAlFilterReport = (req, res) => {

  ModelName.findAllFilterReport(
    req.params.area,
    req.params.road,
    req.params.status,
    (err, data) => {
      if (err) {
        if (err.kind === "not_found") {
          res.status(404).send({
            message: `Not found connection with area id ${req.params.area}.`,
          });
        } else {
          res.status(500).send({
            message:
              "Error retrieving connection with area id " + req.params.area,
          });
        }
      } else res.send(data);
    }
  );
};




// Get Due reports by status
exports.findAllFilterStatusReport = (req, res) => {
  ModelName.findAllFilterStatusReport(req.params.status, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with status id " + req.params.status,
        });
      }
    } else res.send(data);
  });
};

// Get Due reports by min Amount
exports.findAllFilterMinAmountReport = (req, res) => {
  ModelName.findAllFilterMinAmountReport(req.params.amount, (err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with amount id ${req.params.amount}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection  due with amount id " +
            req.params.amount,
        });
      }
    } else res.send(data);
  });
};

exports.findAllUnpaidConnection = (req, res) => {

  // Get unpaid reports by area

  ModelName.findAllUnpaidConnection((err, data) => {
    if (err) {
      if (err.kind === "not_found") {
        res.status(404).send({
          message: `Not found connection with area id ${req.params.area}.`,
        });
      } else {
        res.status(500).send({
          message:
            "Error retrieving connection with area id " + req.params.area,
        });
      }
    } else res.send(data);
  });
};