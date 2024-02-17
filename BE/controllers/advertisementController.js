const express = require('express');
const { Op } = require('sequelize');
const uuid = require('uuid');
const https = require('https');
const router = express.Router();
const Joi = require('joi');
const validateRequest = require('_middleware/validate-request');
var nodemailer = require('nodemailer');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const config = require('../_helpers/config.json');
const verifyToken = require('../_middleware/authorize');
router.post('/', createSchema, verifyToken, createAdvertisemet);
router.get('/', verifyToken, getAllAds);
router.put('/:id', verifyToken, updateAd);
router.delete('/:id', verifyToken, deleteAd);
router.post('/getAdsByDate', verifyToken, getAlladvertisement);
const db = require('../_helpers/dbModel');
module.exports = router;

function createSchema(req, res, next) {
  const schema = Joi.object({
    name: Joi.string().required(),
    date: Joi.date().required(),
    time: Joi.string().required(),
    venue: Joi.string().required(),
    description: Joi.string().required(),
  });
  validateRequest(req, next, schema);
}

async function createAdvertisemet(req, res, next) {
  let { name, date, time, venue, description } = req.body;

  const payload = {
    name,
    date,
    time,
    venue,
    description,
    created_user_Id: req.user.sub.userId,
  };

  // //check user is therer already
  const advertisement = await db.Advertisement.create(payload);
  const getAd = await db.Advertisement.findOne({
    where: {
      id: advertisement?.id,
    },
    include: {
      model: db.User,
    },
  });

  return res.json(getAd);

  //check for index_no
}

async function getAllAds(req, res, next) {
  const userId = req.user.sub.userId;
  try {
    const ads = await db.Advertisement.findAll({
      where: {
        created_user_Id: userId,
      },
    });

    return res.status(200).json({
      data: ads,
    });
  } catch (err) {
    return res.status(500).send('error while fetching advertisements');
  }
}

async function updateAd(req, res, next) {
  const id = req.params.id;
  let { name, date, time, venue, description } = req.body;

  const payload = {
    name,
    date,
    time,
    venue,
    description,
  };
  try {
    const ad = await db.Advertisement.update(payload, {
      where: {
        id,
      },
    });

    return res.status(200).json({
      ad,
    });
  } catch (err) {
    return res.status(500).send('error while updating advertisements');
  }
}

async function deleteAd(req, res, next) {
  const id = req.params.id;

  try {
    await db.Advertisement.destroy({
      where: {
        id,
      },
    });

    return res.status(200).json({
      message: 'Advertisement has been deleted successfully',
    });
  } catch (err) {
    return res.status(500).send('error while deleting advertisements');
  }
}

async function getAlladvertisement(req, res, next) {
  const { date } = req.body;

  try {
    if (!date) {
      const resultsData = await db.Advertisement.findAll({});
      return res.status(200).json(resultsData);
    }

    let startDate = new Date(date);
    const results = await db.Advertisement.findAll({
      where: {
        date: {
          [Op.gte]: new Date(
            startDate.getFullYear(),
            startDate.getMonth(),
            startDate.getDate()
          ),
          [Op.lt]: new Date(
            startDate.getFullYear(),
            startDate.getMonth(),
            startDate.getDate() + 1
          ),
        },
      },
    });
    return res.status(200).json(results);
  } catch (err) {
    console.log(err, 'err');
    return res.status(500).send('error occured');
  }
}
