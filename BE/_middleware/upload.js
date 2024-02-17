// const util = require("util");
// const path = require("path");
// const multer = require("multer");
// var storage = multer.diskStorage({
//     destination: './upload/multifiles',
//     filename: (req, file, cb) => {
//         return cb(null, `${file.fieldname}_${Date.now()}${path.extname(file.originalname)}`)
//     }
//   });
// var uploadFiles = multer({ storage: storage }).array("multifiles", 10);
// var uploadFilesMiddleware = util.promisify(uploadFiles);
// module.exports = uploadFilesMiddleware;
