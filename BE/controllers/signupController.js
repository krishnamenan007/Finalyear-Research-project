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
const { Readable } = require('stream');
const { Blob } = require('buffer');
const multer = require('multer');
const upload = multer();


const db = require('../_helpers/dbModel');
const { hash } = require('bcryptjs');
router.post('/signup', upload.any(), signupSchema, signup);
router.post('/activate', activateUser);
router.post('/signin', signIn);
router.post('/resendLink', resendActivationLink);

module.exports = router;

function bufferToBlob(buffer, type) {
  const blob = new Blob([buffer], { type: type });
  return blob;
}

function signupSchema(req, res, next) {
  const schema = Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().min(8).max(12).required(),
    index_no: Joi.required(),
    year: Joi.optional(),
    role_id: Joi.number(),
    university: Joi.string().required(),
    first_name: Joi.string().required(),
    last_name: Joi.string().required(),
    phone_num1: Joi.string().min(10).max(10).required(),
    phone_num2: Joi.string().optional(),
    department: Joi.string().required(),
  });
  validateRequest(req, next, schema);
}

async function signup(req, res, next) {
  let {

    email,
    index_no,
    phone_num1,
    phone_num2,
    department,
    first_name,
    last_name,
    year,
    password,
    university,
    role_id,
  } = req.body;
  // //check user is there already
  const user = await db.User.findOne({
    where: {
      email: {
        [Op.like]: [`%${email}%`],
      },
    },
  });

  //check for index_no
  if (index_no) {
  }
  const index = await db.User.findOne({
    where: {
      index_no: {
        [Op.like]: [`%${index_no}%`],
      },
    },
  });
  if (user) {
    return res.status(500).json({
      errorMessage: true,
      message: 'user is already there with same email',
      statusCode: 400,
    });
  }
  if (index_no && index) {
    return res.status(500).json({
      errorMessage: true,
      message: 'user with index is already there',
    });
  }
  const pass = await bcrypt.hash(password, 10);
  let fileBuffer = null;
  if (req.files.length) {
    fileBuffer = req.files[0]?.buffer;
  }

  const payload = {
    index_no,
    uuid: uuid.v1(),
    password: pass,
    email,
    year,
    status: 'pending',
    university,
    phone_num1,
    phone_num2,
    first_name: first_name,
    last_name,
    department,
    role_id,
    image: fileBuffer,
  };

  console.log(req.body, 'role_id');

  //create user
  const createdUser = await db.User.create(payload);

  var transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: 'fixit1819@gmail.com',
      pass: 'umsuzlbgxbhdmuhp',
    },
  });

  var mailOptions = {
    from: 'fixit1819@gmail.com',
    to: createdUser.email,
    subject: 'Verify your email via link',
    //   text: 'That was easy!'
    html: `<h1>Congratulations you have successfully registered,Please click on the link to verify</h1><p>link:<a href='http://localhost:3000/user/activation?id=${createdUser.uuid}'>http://localhost:3000/user/activation?id=${createdUser.uuid}</a></p>`,
  };

  transporter.sendMail(mailOptions, function (error, info) {
    if (error) {
      console.log(error);
    } else {
      console.log('Email sent: ' + info.response);
    }
  });

  return res.json({
    statusCode: 200,
    message: 'user created successfully please check  email for verification',
  });
}

async function activateUser(req, res, next) {
  const uuid = req.body.uuid;
  const user = await db.User.findOne({
    where: {
      uuid,
    },
  });
  if (!user) {
    return res.json({
      statusCode: 500,
      message: 'user not found',
    });
  }
  if (user.status === 'active') {
    return res.json({
      statusCode: 203,
      message: 'user is already active',
    });
  }

  await db.User.update(
    {
      status: 'active',
    },
    {
      where: {
        uuid,
      },
    }
  );

  return res.json({
    statusCode: 200,
    message: 'user has been successfully acivated',
  });
}

async function signIn(req, res, next) {
  const password = req.body.password;
  const email = req.body.email;

  let user = await db.User.findOne({
    where: {
      email: {
        [Op.like]: [`%${email}%`],
      },
    },
    include: [
      {
        model: db.Role,
      },
    ],
  });
  if (!user) {
    return res.json({
      statusCode: 401,
      message: 'user not found',
    });
  }
  if (user.status === 'pending') {
    return res.json({
      statusCode: 402,
      message: 'user is not active',
    });
  }

  const isPassTrue = await bcrypt.compare(password, user.password);
  if (!isPassTrue) {
    return res.json({
      statusCode: 403,
      message: 'incorrect pasword',
    });
  }
  user = user.toJSON();

  console.log(user, 'user');

  const gpaDetails = await db.GPA.findOne({
    where: {
      user_id: user.id,
    },
  });

  const token = jwt.sign(
    {
      sub: {
        userId: user.id,
        role_code: user?.role?.role_code,
        first_name: user.last_name,
        is_gpa_submitted: gpaDetails ? true : false,
      },
    },
    config.secret,
    { expiresIn: '7d' }
  );

  return res.json({
    statusCode: 200,
    message: 'login success',
    data: {
      token,
      role_code: user?.role?.role_code,
    },
  });
}
async function resendActivationLink(req, res, next) {
  const email = req.body.email;

  const createdUser = await db.User.findOne({
    where: {
      email: {
        [Op.like]: [`%${email}%`],
      },
    },
  });

  var transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: 'fixit1819@gmail.com',
      pass: 'umsuzlbgxbhdmuhp',
    },
  });

  var mailOptions = {
    from: 'fixit1819@gmail.com',
    to: createdUser.email,
    subject: 'Verify your email via link',
    //   text: 'That was easy!'
    html: `<h1>Congratulations you have successfully registered,Please click on the link to verify</h1><p>link:<a href='http://localhost:3000/user/activation?id=${createdUser.uuid}'>http://localhost:3000/user/activation?id=${createdUser.uuid}</a></p>`,
  };

  transporter.sendMail(mailOptions, function (error, info) {
    if (error) {
      console.log(error);
    } else {
      console.log('Email sent: ' + info.response);
    }
  });

  return res.json({
    statusCode: 200,
    message: 'successfully send the activation link',
  });
}
