const express = require('express');
const { Op } = require('sequelize');
const uuid = require('uuid');
const https = require('https');
const router = express.Router();
const { Readable } = require('stream');
const { Blob } = require('buffer');
const Joi = require('joi');
const validateRequest = require('_middleware/validate-request');
var nodemailer = require('nodemailer');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const config = require('../_helpers/config.json');
const verifyToken = require('../_middleware/authorize');
const multer = require('multer');
const upload = multer();
const db = require('../_helpers/dbModel');

router.post('/', upload.any(), verifyToken, createGPA);

module.exports = router;

async function createGPA(req, res, next) {
  let gpaScore = null;
  let { name } = req.body;

  let file = null;
  if (req.files.length) {
    let fileName = req.files[0]?.originalname;
    let fileBuffer = req.files[0]?.buffer;

    file = await db.File.create({
      file_name: fileName,
      file_data: fileBuffer,
    });
  }

  const gpaDetails = await db.GPA.findOne({
    where: {
      user_id: req.user.sub.userId,
    },
  });

  const skill = await db.Skill.findOne({
    where: {
      user_id: req.user.sub.userId,
    },
  });

  if (req.body.type === 'GPA') {
    if (gpaDetails) {
      const file_id = gpaDetails.file_id;

      await db.GPA.destroy({
        where: {
          id: gpaDetails.id,
        },
      });

      await db.File.destroy({
        where: {
          id: file_id,
        },
      });

      //update knowlege skill

      if (skill) {
        let finalRes = null;
        let toBeDeleted = null;
        let finalResValue = null;
        let { name } = req.body;
        name = parseFloat(name);

        //to be updated
        if (name >= 3.7) {
          finalRes = 25;
        } else if (name >= 3.3 && name < 3.7) {
          finalRes = 20;
        } else if (name >= 3 && name < 3.3) {
          finalRes = 15;
        } else if (name >= 2 && name < 3) {
          finalRes = 10;
        }

        alreadyGpa = gpaDetails?.value;

        //to be removed
        if (alreadyGpa >= 3.7) {
          toBeDeleted = 25;
        } else if (alreadyGpa >= 3.3 && alreadyGpa < 3.7) {
          toBeDeleted = 20;
        } else if (alreadyGpa >= 3 && alreadyGpa < 3.3) {
          toBeDeleted = 15;
        } else if (alreadyGpa >= 2 && alreadyGpa < 3) {
          toBeDeleted = 10;
        } else {
          toBeDeleted = 0;
        }

        if (skill.knowledge) {
          let alreadyExisting = skill.knowledge
            ? parseFloat(skill.knowledge)
            : 0;
          finalResValue = alreadyExisting + finalRes - toBeDeleted;
        } else {
          finalResValue = finalRes;
        }

        await db.Skill.update(
          {
            knowledge: finalResValue,
          },
          {
            where: {
              user_id: req.user.sub.userId,
            },
          }
        );
      }
    }
    const payload = {
      value: name,
      file_id: file ? file.id : null,
      user_id: req.user.sub.userId,
    };

    const gpaFloat = parseFloat(name);

    console.log(gpaFloat, 'rtrt');

    if (gpaFloat >= 3.7) {
      gpaScore = 25;
    } else if (gpaFloat >= 3.3 && gpaFloat < 3.7) {
      gpaScore = 20;
    } else if (gpaFloat >= 3 && gpaFloat < 3.3) {
      gpaScore = 15;
    } else if (gpaFloat >= 2 && gpaFloat < 3) {
      gpaScore = 10;
    }

    console.log(gpaScore, 'rtrt');

    try {
      const gpa = await db.GPA.create(payload);

      const gpaData = await db.GPA.findOne({
        where: {
          id: gpa?.id,
        },
        include: [
          {
            model: db.File,
          },
        ],
      });
      if (!skill) {
        const payloadValueGpa = {
          knowledge: gpaScore,
          user_id: req.user.sub.userId,
        };
        await db.Skill.create(payloadValueGpa);
      }
      return res.status(200).json(gpaData);
    } catch (err) {
      console.log(err);
      return res.status(500).send('error while creating  advertisements');
    }
  }
  else if (req.body.type === 'SELF_LEARNING') {
    const { course_name, course_level, certificate_no } = req.body;
    if (certificate_no) {
    }
    const certificate = await db.SelfLearning.findOne({
      where: {
        certificate_no: {
          [Op.like]: [`%${certificate_no}%`],
        },
      },
    });
   
    if (certificate_no && certificate) {
      return res.status(500).json({
        errorMessage: true,
        message: 'certificate with number is already there',
      });
    }
    const payload = {
      course_name,
      course_level,
      certificate_no,
      file_id: file ? file.id : null,
      user_id: req.user.sub.userId,
      approved: false,
      rejected: false
    };
    try {
      const selfLearning = await db.SelfLearning.create(payload);
      const selfLearningData = await db.SelfLearning.findOne({
        where: {
          id: selfLearning?.id,
        },
        include: [
          {
            model: db.File,
          },
        ],
      });
      return res.status(200).json(selfLearningData);
    } catch (err) {
      console.log(err);
      return res.status(500).send('error while creating selflearning');
    }
  } else if (req.body.type === 'CLUB') {
    const { club_name, certificate_no, club_level } = req.body;

    if (certificate_no) {
    }
    const certificate = await db.Club.findOne({
      where: {
        certificate_no: {
          [Op.like]: [`%${certificate_no}%`],
        },
      },
    });
   
    if (certificate_no && certificate) {
      return res.status(500).json({
        errorMessage: true,
        message: 'certificate with number is already there',
      });
    }


    const payload = {
      club_name,
      certificate_no,
      club_level,
      file_id: file ? file.id : null,
      user_id: req.user.sub.userId,
    };
    try {
      const club = await db.Club.create(payload);

      const clubData = await db.Club.findOne({
        where: {
          id: club?.id,
        },
        include: [
          {
            model: db.File,
          },
        ],
      });

      let communication = null;
      let leadership = null;
      let team_work = null;
      let descision_making = null;

      if (club_level === 'president') {
        communication = 5;
        leadership = 5;
        descision_making = 2;
        team_work = 3;
      } else if (club_level === 'lead') {
        communication = 4;
        leadership = 3;
        descision_making = 1;
        team_work = 2;
      } else if (club_level === 'member') {
        communication = 3;
        leadership = 0;
        descision_making = 1;
        team_work = 1;
      }
      if (skill) {
        communication = communication + skill?.communication;
        leadership = leadership + skill?.leadership;
        descision_making = descision_making + skill?.descision_making;
        team_work = team_work + skill?.team_work;

        await db.Skill.update(
          {
            communication,
            leadership,
            descision_making,
            team_work,
          },
          {
            where: {
              user_id: req?.user?.sub.userId,
            },
          }
        );
      } else {
        await db.Skill.create({
          communication,
          leadership,
          descision_making,
          team_work,
          user_id: req.user.sub.userId,
        });
      }
      return res.status(200).json(clubData);
    } catch (err) {
      console.log(err);
      return res.status(500).send('error while creating club details');
    }
  } else if (req.body.type === 'SPORT') {
    const { sport_name, certificate_no, level, cardinality, position } = req.body;
    if (certificate_no) {
    }
    const certificate = await db.Sport.findOne({
      where: {
        certificate_no: {
          [Op.like]: [`%${certificate_no}%`],
        },
      },
    });
   
    if (certificate_no && certificate) {
      return res.status(500).json({
        errorMessage: true,
        message: 'certificate with number is already there',
      });
    }
    
    const payload = {
      sport_name,
      certificate_no,
      level,
      cardinality,
      position,
      file_id: file ? file.id : null,
      user_id: req.user.sub.userId,
      approved: false,
      rejected: false
    };
    try {
      const sport = await db.Sport.create(payload);
  
      const sportData = await db.Sport.findOne({
        where: {
          id: sport?.id,
        },
        include: [
          {
            model: db.File,
          },
        ],
      });
      return res.status(200).json(sportData);
    } catch (err) {
      console.log(err);
      return res.status(500).send('error while creating sport details');
    }
  }
   else if (req.body.type === 'EVENT') {
    const { event_name, certificate_no, level, cardinality, position } = req.body;
    
    if (certificate_no) {
    }
    const certificate = await db.Event.findOne({
      where: {
        certificate_no: {
          [Op.like]: [`%${certificate_no}%`],
        },
      },
    });
   
    if (certificate_no && certificate) {
      return res.status(500).json({
        errorMessage: true,
        message: 'certificate with number is already there',
      });
    }
    const payload = {
      event_name,
      certificate_no,
      level,
      cardinality,
      position,
      file_id: file ? file.id : null,
      user_id: req.user.sub.userId,
    };
    try {
      const sport = await db.Event.create(payload);

      const sportData = await db.Event.findOne({
        where: {
          id: sport?.id,
        },
        include: [
          {
            model: db.File,
          },
        ],
      });

      let teamwork = null;
      let problem_solving = null;
      let creativity = null;
      let leadership = null;
      let descision_making = null;

      if (level === 'university') {
        if (position === '1') {
          if (cardinality === 'leader') {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === 'member') {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 4;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === '2') {
          if (cardinality === 'leader') {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === 'member') {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === '3') {
          if (cardinality === 'leader') {
            teamwork = 1;
            leadership = 1;
            problem_solving = 1;
            creativity = 1;
            descision_making = 1;
          }
          if (cardinality === 'member') {
            teamwork = 2;
            problem_solving = 1;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 3;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === 'part') {
          if (cardinality === 'leader') {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === 'member') {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === 'national') {
        if (position === '1') {
          if (cardinality === 'leader') {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === 'member') {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === '2') {
          if (cardinality === 'leader') {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === 'member') {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === '3') {
          if (cardinality === 'leader') {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === 'member') {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === 'part') {
          if (cardinality === 'leader') {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === 'member') {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === 'international') {
        if (position === '1') {
          if (cardinality === 'leader') {
            teamwork = 5;
            leadership = 5;
            problem_solving = 5;
            creativity = 5;
            descision_making = 5;
          }
          if (cardinality === 'member') {
            teamwork = 10;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 15;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === '2') {
          if (cardinality === 'leader') {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === 'member') {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === '3') {
          if (cardinality === 'leader') {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === 'member') {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === 'part') {
          if (cardinality === 'leader') {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === 'member') {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === 'individual') {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      }

      if (skill) {
        problem_solving = problem_solving + skill?.problem_solving;
        descision_making = descision_making + skill?.descision_making;
        teamwork = teamwork + skill?.team_work;
        leadership = leadership + skill?.leadership;

        await db.Skill.update(
          {
            problem_solving,
            leadership,
            descision_making,
            team_work: teamwork,
            creativity: creativity,
          },
          {
            where: {
              user_id: req?.user?.sub.userId,
            },
          }
        );
      } else {
        await db.Skill.create({
          problem_solving,
          leadership,
          descision_making,
          team_work: teamwork,
          creativity,
          user_id: req.user.sub.userId,
        });
      }
      return res.status(200).json(sportData);
    } catch (err) {
      console.log(err);
      return res.status(500).send('error while creating sport details');
    }
  }
}
