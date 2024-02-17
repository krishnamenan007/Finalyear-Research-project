const jwt = require('jsonwebtoken');

const { secret } = require('../_helpers/config.json');

const verifyToken = (req, res, next) => {
  let token = req.body.token || req.query.token || req.headers.authorization;
  // console.log(req.headers.authorization, 'etetetet');
  if (!token) {
    return res.status(403).send('A token is required for authentication');
  }
  token = req.headers.authorization;
  token = token.replace('Bearer ', '');

  try {
    const decoded = jwt.verify(token, secret);
    req.user = decoded;
  } catch (err) {
    return res.status(401).send('Invalid Token');
  }
  return next();
};

module.exports = verifyToken;
