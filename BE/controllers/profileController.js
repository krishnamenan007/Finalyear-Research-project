const express = require("express");
const router = express.Router();
const verifyToken = require("../_middleware/authorize");
const db = require("../_helpers/dbModel");
const { Readable } = require("stream");
const { Blob } = require("buffer");
const multer = require("multer");
const config = require("../_helpers/config.json");
const { Sequelize, QueryTypes } = require("sequelize");

const upload = multer();
router.get("/", verifyToken, getProfileDetails);
router.put("/updatePic", upload.any(), verifyToken, updateProfilePic);

router.put("/updateInfo", verifyToken, upadteInfo);

router.put("/updateStudentDetails", verifyToken, updateStudentData);
router.put("/deleteStuInfo", verifyToken, deleteStudInfo);

module.exports = router;

async function getProfileDetails(req, res, next) {
  const userID = req.user.sub.userId;
  const { host, port, user, password, database } = config.database;

  const sequelize = new Sequelize(database, user, password, {
    dialect: "mysql",
  });

//   const countQuery = `SELECT
//   u.id,u.status,u.index_no,
//   u.first_name AS 'firstName',
//   u.last_name AS 'lastName',
//   u.index_no AS 'indexNo',
//   u.role_id,
//   COALESCE(user_scores.total_score, 0) AS total_score,
//   user_scores.ranks
// FROM
//   users u
// LEFT JOIN (
//   SELECT
//       u.id,
//       SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) AS total_score,
//       RANK() OVER (ORDER BY SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) DESC) AS 'ranks'
//   FROM
//       users u
//   JOIN
//       skills s ON u.id = s.user_id
//   GROUP BY
//       u.id
// ) AS user_scores ON user_scores.id = u.id
// where status like 'active' and role_id = 1 
// order by total_score DESC
// `;
  
// let results = await sequelize.query(countQuery, {
//     type: QueryTypes.SELECT,
//   });
const countQuery = `
SELECT
  u.id,
  u.status,
  u.index_no,
  u.first_name AS 'firstName',
  u.last_name AS 'lastName',
  u.index_no AS 'indexNo',
  u.role_id,
  COALESCE(user_scores.total_score, 0) AS total_score,
  user_scores.ranks,
  s.performance_level
FROM
  users u
LEFT JOIN (
  SELECT
      u.id,
      SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) AS total_score,
      RANK() OVER (ORDER BY SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) DESC) AS 'ranks'
  FROM
      users u
  JOIN
      skills s ON u.id = s.user_id
  GROUP BY
      u.id
) AS user_scores ON user_scores.id = u.id
LEFT JOIN
  skills s ON u.id = s.user_id
WHERE
  u.status = 'active'
  AND u.role_id = 1
ORDER BY
  total_score DESC
`;

let results = await sequelize.query(countQuery, {
  type: QueryTypes.SELECT,
});



  results = results.find((res) => res.id === userID);

  try {
    let profileDetails = await db.User.findOne({
      attributes: { exclude: ["password", "uuid"] },
      where: {
        id: userID,
      },
      include: [
        {
          model: db.Club,
        },
        {
          model: db.SelfLearning,
        },
        {
          model: db.GPA,
        },
        {
          model: db.Skill,
        },
        {
          model: db.Role,
        },
        {
          model: db.Event,
        },
        {
          model: db.Sport,
        },
      ],
    });

    profileDetails = profileDetails.toJSON();

    profileDetails = {
      ...profileDetails,
      rank: results?.ranks || 0,
    };

    console.log(profileDetails, "vv");
    return res.status(200).json(profileDetails);
  } catch (error) {
    console.log(error, "vv");

    return res.status(500).send("error while fetching profile details");
  }
}

async function updateProfilePic(req, res, next) {
  let fileBuffer = null;

  if (req.files.length) {
    fileBuffer = req.files[0]?.buffer;
  }

  const userId = req.user.sub.userId;
  try {
    await db.User.update(
      {
        image: fileBuffer,
      },
      {
        where: {
          id: userId,
        },
      }
    );
    return res.status(200).send("profilepic is updated Successfully");
  } catch (error) {
    console.log(error);
    return res.status(500).send("error while updating profile details");
  }
}

async function upadteInfo(req, res, next) {
  const userId = req.user.sub.userId;
  const {
    first_name,
    last_name,
    phone_num1,
    phone_num2,
    university,
    department,
    year,
    index_no,
  } = req.body;
  const payload = {
    first_name,
    last_name,
    phone_num1,
    phone_num2,
    university,
    year,
    index_no,
    department,
  };

  try {
    const updatedInfo = await db.User.update(payload, {
      where: {
        id: userId,
      },
    });

    return res.status(200).json(updatedInfo);
  } catch (error) {
    return res.status(500).send("error when updating profile");
  }
}

async function updateStudentData(req, res, next) {
  try {
    let skill = await db.Skill.findOne({
      where: {
        user_id: req.user.sub.userId,
      },
    });
    if (req.body.type === "CLUBS") {
      const { club_level, certificate_no,club_name, id } = req.body;

      let communication;
      let leadership;
      let descision_making;
      let team_work;

      let communicationDeleted;
      let leadershipDeleted;
      let descision_makingDeleted;
      let team_workDeleted;

      let clubDetails = await db.Club.findOne({
        where: {
          id,
        },
      });

      if (club_level === "president") {
        communication = 5;
        leadership = 5;
        descision_making = 2;
        team_work = 3;
      } else if (club_level === "lead") {
        communication = 4;
        leadership = 3;
        descision_making = 1;
        team_work = 2;
      } else if (club_level === "member") {
        communication = 3;
        leadership = 0;
        descision_making = 1;
        team_work = 1;
      }

      if (clubDetails.club_level === "president") {
        communicationDeleted = 5;
        leadershipDeleted = 5;
        descision_makingDeleted = 2;
        team_workDeleted = 3;
      } else if (clubDetails.club_level === "lead") {
        communicationDeleted = 4;
        leadershipDeleted = 3;
        descision_makingDeleted = 1;
        team_workDeleted = 2;
      } else if (clubDetails.club_level === "member") {
        communicationDeleted = 3;
        leadershipDeleted = 0;
        descision_makingDeleted = 1;
        team_workDeleted = 1;
      }

      let comVal = skill?.communication ? parseFloat(skill?.communication) : 0;
      let leadVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let dsVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;
      let teamVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;

      const payload = {
        communication: comVal - communicationDeleted + communication,
        leadership: leadVal - leadershipDeleted + leadership,
        team_work: dsVal - team_workDeleted + team_work,
        descision_making: teamVal - descision_makingDeleted + descision_making,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Club.update(
        {
          club_name,
          certificate_no,
          club_level,
        },
        {
          where: {
            id,
          },
        }
      );
    } else if (req.body.type === "SPORT") {
      const { id, cardinality, level,certificate_no, sport_name, position } = req.body;

      const res = await db.Sport.findOne({
        where: {
          id,
        },
      });

      console.log(res, "res");

      const resltsToBeDeleted = await getdeletedSportVal(
        res.level,
        res.certificate_no,
        res.position,
        res.cardinality
      );

      const resultsToBeAdded = await getdeletedSportVal(
        level,
        certificate_no,
        position,
        cardinality
      );

      let desVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;
      let leadVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let teamVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;

      const payload = {
        team_work:
          teamVal - resltsToBeDeleted.teamwork + resultsToBeAdded.teamwork,
        descision_making:
          desVal -
          resltsToBeDeleted.descision_making +
          resultsToBeAdded.descision_making,
        leadership:
          leadVal - resltsToBeDeleted.leadership + resultsToBeAdded.leadership,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Sport.update(
        { sport_name, certificate_no, level, cardinality, position },
        {
          where: {
            id,
          },
        }
      );
    } else if (req.body.type === "EVENT") {
      const { position, cardinality, level, certificate_no, event_name, id } = req.body;

      let teamwork = null;
      let problem_solving = null;
      let creativity = null;
      let leadership = null;
      let descision_making = null;

      let teamworkDeleted = null;
      let problem_solvingDeleted = null;
      let creativityDeleted = null;
      let leadershipDeleted = null;
      let descision_makingDeleted = null;

      let eventData = await db.Event.findOne({
        where: {
          id,
        },
      });

      //tobe added

      if (level === "university") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 4;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === "member") {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 1;
            problem_solving = 1;
            creativity = 1;
            descision_making = 1;
          }
          if (cardinality === "member") {
            teamwork = 2;
            problem_solving = 1;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 3;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === "national") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === "member") {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === "member") {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === "international") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 5;
            leadership = 5;
            problem_solving = 5;
            creativity = 5;
            descision_making = 5;
          }
          if (cardinality === "member") {
            teamwork = 10;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 15;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === "member") {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      }

      //tobeDeleted

      if (eventData?.level === "university") {
        if (eventData?.position === "1") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 3;
            leadershipDeleted = 3;
            problem_solvingDeleted = 3;
            creativityDeleted = 3;
            descision_makingDeleted = 3;
          }
          if (eventData?.cardinality === "member") {
            teamwork = 4;
            problem_solvingDeleted = 4;
            descision_makingDeleted = 4;
            creativityDeleted = 4;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }
        if (eventData?.position === "2") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 2;
            leadershipDeleted = 2;
            problem_solvingDeleted = 2;
            creativityDeleted = 2;
            descision_makingDeleted = 2;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 3;
            problem_solvingDeleted = 3;
            descision_makingDeleted = 2;
            creativityDeleted = 2;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 5;
            descision_makingDeleted = 3;
            creativityDeleted = 2;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "3") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 1;
            leadershipDeleted = 1;
            problem_solvingDeleted = 1;
            creativityDeleted = 1;
            descision_makingDeleted = 1;
          }
          if (eventData?.cardinality === "member") {
            teamwork = 2;
            problem_solvingDeleted = 1;
            descision_makingDeleted = 1;
            creativityDeleted = 1;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 3;
            descision_makingDeleted = 1;
            creativityDeleted = 1;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "part") {
          if (cardinality === "leader") {
            teamworkDeleted = 0.5;
            leadershipDeleted = 0.5;
            problem_solvingDeleted = 0.5;
            creativityDeleted = 0.5;
            descision_makingDeleted = 0.5;
          }
          if (eventData?.cardinality === "member") {
            teamwork = 1;
            problem_solvingDeleted = 0.5;
            descision_makingDeleted = 0.5;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 1;
            descision_makingDeleted = 1;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }
      } else if (eventData?.level === "national") {
        if (eventData?.position === "1") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 4;
            leadershipDeleted = 4;
            problem_solvingDeleted = 4;
            creativityDeleted = 4;
            descision_makingDeleted = 4;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 5;
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 10;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "2") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 3;
            leadershipDeleted = 3;
            problem_solvingDeleted = 3;
            creativityDeleted = 3;
            descision_makingDeleted = 3;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 4;
            problem_solvingDeleted = 4;
            descision_makingDeleted = 4;
            creativityDeleted = 3;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "3") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 2;
            leadershipDeleted = 2;
            problem_solvingDeleted = 2;
            creativityDeleted = 2;
            descision_makingDeleted = 2;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 3;
            problem_solvingDeleted = 3;
            descision_makingDeleted = 2;
            creativityDeleted = 2;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 5;
            descision_makingDeleted = 3;
            creativityDeleted = 2;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "part") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 0.5;
            leadershipDeleted = 0.5;
            problem_solvingDeleted = 0.5;
            creativityDeleted = 0.5;
            descision_makingDeleted = 0.5;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 1;
            problem_solvingDeleted = 0.5;
            descision_makingDeleted = 0.5;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 1;
            descision_makingDeleted = 1;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }
      } else if (eventData?.level === "international") {
        if (eventData?.position === "1") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 5;
            leadershipDeleted = 5;
            problem_solvingDeleted = 5;
            creativityDeleted = 5;
            descision_makingDeleted = 5;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 10;
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 15;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }
        if (eventData?.position === "2") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 4;
            leadershipDeleted = 4;
            problem_solvingDeleted = 4;
            creativityDeleted = 4;
            descision_makingDeleted = 4;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 5;
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 10;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "3") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 3;
            leadershipDeleted = 3;
            problem_solvingDeleted = 3;
            creativityDeleted = 3;
            descision_makingDeleted = 3;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 4;
            problem_solvingDeleted = 4;
            descision_makingDeleted = 4;
            creativityDeleted = 3;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 5;
            descision_makingDeleted = 5;
            creativityDeleted = 5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }

        if (eventData?.position === "part") {
          if (eventData?.cardinality === "leader") {
            teamworkDeleted = 0.5;
            leadershipDeleted = 0.5;
            problem_solvingDeleted = 0.5;
            creativityDeleted = 0.5;
            descision_makingDeleted = 0.5;
          }
          if (eventData?.cardinality === "member") {
            teamworkDeleted = 1;
            problem_solvingDeleted = 0.5;
            descision_makingDeleted = 0.5;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
          }
          if (eventData?.cardinality === "individual") {
            problem_solvingDeleted = 1;
            descision_makingDeleted = 1;
            creativityDeleted = 0.5;
            leadershipDeleted = 0;
            teamworkDeleted = 0;
          }
        }
      }

      let problem_solvingVal = skill?.problem_solving
        ? parseFloat(skill?.problem_solving)
        : 0;
      let creativity_val = skill?.creativity
        ? parseFloat(skill?.creativity)
        : 0;
      let leadershipVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let teamWorkVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;
      let desMakingVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;

      const data = {
        leadershipVal,
        leadershipDeleted,
        leadership,
        desMakingVal,
        descision_makingDeleted,
        descision_making,
        teamWorkVal,
        teamworkDeleted,
        teamwork,
        creativity,
        creativity_val,
        creativityDeleted,
        problem_solving,
        problem_solvingDeleted,
        problem_solvingVal,
      };

      console.log(data, "lll");

      const payload = {
        leadership: leadershipVal - leadershipDeleted + leadership,
        descision_making:
          desMakingVal - descision_makingDeleted + descision_making,
        team_work: teamWorkVal - teamworkDeleted + teamwork,
        problem_solving:
          problem_solvingVal - problem_solvingDeleted + problem_solving,
        creativity: creativity_val - creativityDeleted + creativity,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Event.update(
        {
          event_name,
          certificate_no,
          level,
          cardinality,
          position,
        },
        {
          where: {
            id,
          },
        }
      );
    } else if (req.body.type === "SELF_LEARNINGS") {
      const { certificate_no,course_level, course_name, id } = req.body;

      let knowledgeSkill = null;
      let psSkill = null;
      let creativity = null;

      let KnowDeleted;
      let psSkillDeleted;
      let creativityDeleted;

      let selfLearningDetails = await db.SelfLearning.findOne({
        where: {
          id,
        },
      });
      if (course_level === "beginner") {
        knowledgeSkill = 5;
        (psSkill = 5), (creativity = 5);
      } else if (course_level === "intermediate") {
        knowledgeSkill = 10;
        (psSkill = 5), (creativity = 5);
      } else if (course_level === "expert") {
        knowledgeSkill = 10;
        (psSkill = 10), (creativity = 5);
      }

      if (selfLearningDetails.course_level === "beginner") {
        KnowDeleted = 5;
        (psSkillDeleted = 5), (creativityDeleted = 5);
      } else if (selfLearningDetails.course_level === "intermediate") {
        KnowDeleted = 10;
        (psSkillDeleted = 5), (creativityDeleted = 5);
      } else if (selfLearningDetails.course_level === "expert") {
        KnowDeleted = 10;
        (psSkillDeleted = 10), (creativityDeleted = 5);
      }

      let knowVal = skill?.knowledge ? parseFloat(skill?.knowledge) : 0;
      let psVal = skill?.problem_solving
        ? parseFloat(skill?.problem_solving)
        : 0;
      let creatVal = skill?.creativity ? parseFloat(skill?.creativity) : 0;

      const payload = {
        knowledge: knowVal - KnowDeleted + knowledgeSkill,
        problem_solving: psVal - psSkillDeleted + psSkill,
        creativity: creatVal - creativityDeleted + creativity,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.SelfLearning.update(
        {
          certificate_no,
          course_level,
          course_name,
        },
        {
          where: {
            id,
          },
        }
      );
    }

    return res.status(200).send("succefully  updated the  profile ");
  } catch (error) {
    console.log(error);
    return res.status(500).send("error when updating profile ");
  }
}

async function deleteStudInfo(req, res, next) {
  try {
    let skill = await db.Skill.findOne({
      where: {
        user_id: req.user.sub.userId,
      },
    });
    if (req.body.type === "CLUBS") {
      const { id } = req.body;

      let communicationDeleted;
      let leadershipDeleted;
      let descision_makingDeleted;
      let team_workDeleted;

      let clubDetails = await db.Club.findOne({
        where: {
          id,
        },
      });

      if (clubDetails.club_level === "president") {
        communicationDeleted = 5;
        leadershipDeleted = 5;
        descision_makingDeleted = 2;
        team_workDeleted = 3;
      } else if (clubDetails.club_level === "lead") {
        communicationDeleted = 4;
        leadershipDeleted = 3;
        descision_makingDeleted = 1;
        team_workDeleted = 2;
      } else if (clubDetails.club_level === "member") {
        communicationDeleted = 3;
        leadershipDeleted = 0;
        descision_makingDeleted = 1;
        team_workDeleted = 1;
      }

      let comVal = skill?.communication ? parseFloat(skill?.communication) : 0;
      let leadVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let dsVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;
      let teamVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;

      const payload = {
        communication: comVal - communicationDeleted,
        leadership: leadVal - leadershipDeleted,
        team_work: teamVal - team_workDeleted,
        descision_making: dsVal - descision_makingDeleted,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Club.destroy({
        where: {
          id,
        },
      });
    } else if (req.body.type === "SELF_LEARNINGS") {
      const { id } = req.body;

      let knowledgeSkill = 0;
      let creativity = 0;
      let psSkill = 0;

      let selfLearningData = await db.SelfLearning.findOne({
        where: {
          id,
        },
      });

      if (selfLearningData?.course_level === "beginner") {
        knowledgeSkill = 5;
        (psSkill = 5), (creativity = 5);
      } else if (selfLearningData?.course_level === "intermediate") {
        knowledgeSkill = 10;
        (psSkill = 5), (creativity = 5);
      } else if (selfLearningData?.course_level === "expert") {
        knowledgeSkill = 10;
        (psSkill = 10), (creativity = 5);
      }

      let knowVal = skill?.knowledge ? parseFloat(skill?.knowledge) : 0;
      let psVal = skill?.problem_solving
        ? parseFloat(skill?.problem_solving)
        : 0;
      let creatVal = skill?.creativity ? parseFloat(skill?.creativity) : 0;

      const payload = {
        creativity: creatVal - creativity,
        problem_solving: psVal - psSkill,
        knowledge: knowVal - knowledgeSkill,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.SelfLearning.destroy({
        where: {
          id,
        },
      });
    } else if (req.body.type === "EVENT") {
      const { id } = req.body;

      const { level, position, cardinality } = await db.Event.findOne({
        where: {
          id,
        },
      });

      let teamwork = 0;
      let problem_solving = 0;
      let creativity = 0;
      let leadership = 0;
      let descision_making = 0;

      if (level === "university") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 4;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === "member") {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 1;
            problem_solving = 1;
            creativity = 1;
            descision_making = 1;
          }
          if (cardinality === "member") {
            teamwork = 2;
            problem_solving = 1;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 3;
            descision_making = 1;
            creativity = 1;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === "national") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === "member") {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            problem_solving = 2;
            creativity = 2;
            descision_making = 2;
          }
          if (cardinality === "member") {
            teamwork = 3;
            problem_solving = 3;
            descision_making = 2;
            creativity = 2;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 3;
            creativity = 2;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      } else if (level === "international") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 5;
            leadership = 5;
            problem_solving = 5;
            creativity = 5;
            descision_making = 5;
          }
          if (cardinality === "member") {
            teamwork = 10;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 15;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 4;
            leadership = 4;
            problem_solving = 4;
            creativity = 4;
            descision_making = 4;
          }
          if (cardinality === "member") {
            teamwork = 5;
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 10;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 3;
            leadership = 3;
            problem_solving = 3;
            creativity = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            problem_solving = 4;
            descision_making = 4;
            creativity = 3;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 5;
            descision_making = 5;
            creativity = 5;
            leadership = 0;
            teamwork = 0;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 0.5;
            leadership = 0.5;
            problem_solving = 0.5;
            creativity = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            problem_solving = 0.5;
            descision_making = 0.5;
            creativity = 0.5;
            leadership = 0;
          }
          if (cardinality === "individual") {
            problem_solving = 1;
            descision_making = 1;
            creativity = 0.5;
            leadership = 0;
            teamwork = 0;
          }
        }
      }

      let problem_solvingVal = skill?.problem_solving
        ? parseFloat(skill?.problem_solving)
        : 0;
      let desVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;
      let creatVal = skill?.creativity ? parseFloat(skill?.creativity) : 0;
      let leadVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let teamVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;

      const payload = {
        creativity: creatVal - creativity,
        problem_solving: problem_solvingVal - problem_solving,
        team_work: teamVal - teamwork,
        descision_making: desVal - descision_making,
        leadership: leadVal - leadership,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Event.destroy({
        where: {
          id,
        },
      });
    } else if (req.body.type === "SPORT") {
      const { id } = req.body;

      const { certificate_no, level, position, cardinality } = await db.Sport.findOne({
        where: {
          id,
        },
      });

      let teamwork = 0;
      let leadership = 0;
      let descision_making = 0;

      if (level === "university") {
        if (position === "1") {
          if (cardinality === "leader") {
            descision_making = 3;
            leadership = 2;
            teamwork = 2;
          }
          if (cardinality === "member") {
            descision_making = 3;
            leadership = 0;
            teamwork = 4;
          }
          if (cardinality === "individual") {
            descision_making = 7;
            leadership = 0;
            teamwork = 0;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 2;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 3;
            leadership = 0;
            descision_making = 3;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 6;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 1;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 3;
            leadership = 0;
            descision_making = 2;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 5;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            leadership = 0;
            descision_making = 1;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 2;
          }
        }
      } else if (level === "national") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 3;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            leadership = 0;
            descision_making = 4;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 8;
          }
        }

        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            leadership = 0;
            descision_making = 3;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 7;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 2;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 3;
            leadership = 0;
            descision_making = 3;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 6;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            leadership = 0;
            descision_making = 1;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 2;
          }
        }
      } else if (level === "international") {
        if (position === "1") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 3;
            descision_making = 5;
          }
          if (cardinality === "member") {
            teamwork = 5;
            leadership = 0;
            descision_making = 5;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 10;
          }
        }
        if (position === "2") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            leadership = 0;
            descision_making = 4;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 8;
          }
        }

        if (position === "3") {
          if (cardinality === "leader") {
            teamwork = 2;
            leadership = 2;
            descision_making = 3;
          }
          if (cardinality === "member") {
            teamwork = 4;
            leadership = 0;
            descision_making = 3;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 7;
          }
        }

        if (position === "part") {
          if (cardinality === "leader") {
            teamwork = 1;
            leadership = 0.5;
            descision_making = 0.5;
          }
          if (cardinality === "member") {
            teamwork = 1;
            leadership = 0;
            descision_making = 1;
          }
          if (cardinality === "individual") {
            teamwork = 0;
            leadership = 0;
            descision_making = 2;
          }
        }
      }

      let desVal = skill?.descision_making
        ? parseFloat(skill?.descision_making)
        : 0;
      let leadVal = skill?.leadership ? parseFloat(skill?.leadership) : 0;
      let teamVal = skill?.team_work ? parseFloat(skill?.team_work) : 0;

      const payload = {
        team_work: teamVal - teamwork,
        descision_making: desVal - descision_making,
        leadership: leadVal - leadership,
      };

      await db.Skill.update(payload, {
        where: {
          user_id: req.user.sub.userId,
        },
      });
      await db.Sport.destroy({
        where: {
          id,
        },
      });
    }

    return res.status(200).send("update profile is completed");
  } catch (error) {
    console.log(error, "err");
    return res.status(500).send("error when updating profile ");
  }
}

async function getdeletedSportVal(level, position, cardinality) {
  let teamwork = 0;
  let leadership = 0;
  let descision_making = 0;

  if (level === "university") {
    if (position === "1") {
      if (cardinality === "leader") {
        descision_making = 3;
        leadership = 2;
        teamwork = 2;
      }
      if (cardinality === "member") {
        descision_making = 3;
        leadership = 0;
        teamwork = 4;
      }
      if (cardinality === "individual") {
        descision_making = 7;
        leadership = 0;
        teamwork = 0;
      }
    }
    if (position === "2") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 2;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 3;
        leadership = 0;
        descision_making = 3;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 6;
      }
    }

    if (position === "3") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 1;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 3;
        leadership = 0;
        descision_making = 2;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 5;
      }
    }

    if (position === "part") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 0.5;
        descision_making = 0.5;
      }
      if (cardinality === "member") {
        teamwork = 1;
        leadership = 0;
        descision_making = 1;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 2;
      }
    }
  } else if (level === "national") {
    if (position === "1") {
      if (cardinality === "leader") {
        teamwork = 2;
        leadership = 3;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 4;
        leadership = 0;
        descision_making = 4;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 8;
      }
    }

    if (position === "2") {
      if (cardinality === "leader") {
        teamwork = 2;
        leadership = 2;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 4;
        leadership = 0;
        descision_making = 3;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 7;
      }
    }

    if (position === "3") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 2;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 3;
        leadership = 0;
        descision_making = 3;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 6;
      }
    }

    if (position === "part") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 0.5;
        descision_making = 0.5;
      }
      if (cardinality === "member") {
        teamwork = 1;
        leadership = 0;
        descision_making = 1;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 2;
      }
    }
  } else if (level === "international") {
    if (position === "1") {
      if (cardinality === "leader") {
        teamwork = 2;
        leadership = 3;
        descision_making = 5;
      }
      if (cardinality === "member") {
        teamwork = 5;
        leadership = 0;
        descision_making = 5;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 10;
      }
    }
    if (position === "2") {
      if (cardinality === "leader") {
        teamwork = 2;
        leadership = 3;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 4;
        leadership = 0;
        descision_making = 4;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 8;
      }
    }

    if (position === "3") {
      if (cardinality === "leader") {
        teamwork = 2;
        leadership = 2;
        descision_making = 3;
      }
      if (cardinality === "member") {
        teamwork = 4;
        leadership = 0;
        descision_making = 3;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 7;
      }
    }

    if (position === "part") {
      if (cardinality === "leader") {
        teamwork = 1;
        leadership = 0.5;
        descision_making = 0.5;
      }
      if (cardinality === "member") {
        teamwork = 1;
        leadership = 0;
        descision_making = 1;
      }
      if (cardinality === "individual") {
        teamwork = 0;
        leadership = 0;
        descision_making = 2;
      }
    }
  }

  return {
    teamwork,
    leadership,
    descision_making,
  };
}
