const express = require('express');
const router = express.Router();
const verifyToken = require('../_middleware/authorize');
const db = require('../_helpers/dbModel');
const { Sequelize, QueryTypes } = require('sequelize');
const config = require('../_helpers/config.json');

router.get('/', verifyToken, getAllStudents);

module.exports = router;

async function getAllStudents(req, res, next) {
  const { page, limit, search } = req.query;

  const { host, port, user, password, database } = config.database;

  const sequelize = new Sequelize(database, user, password, {
    dialect: 'mysql',
  });

  const offset = (page - 1) * limit;

  const query = `SELECT
                u.id,u.status,u.index_no,
                u.first_name AS 'firstName',
                u.last_name AS 'lastName',
                u.index_no AS 'indexNo',
                u.role_id,
                user_scores.performance_level,
                COALESCE(user_scores.total_score, 0) AS total_score,
                user_scores.ranks
              
            FROM
                users u
            LEFT JOIN (
                SELECT
                    u.id,
                    s.performance_level,
                    SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) AS total_score,
                    RANK() OVER (ORDER BY SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) DESC) AS 'ranks'
                FROM
                    users u
                JOIN
                    skills s ON u.id = s.user_id
                  
                GROUP BY
                    u.id
            ) AS user_scores ON user_scores.id = u.id
            where status like 'active' and role_id = 1 and index_no like '%${search}%'
            order by total_score DESC
            limit ${limit} OFFSET ${offset}`;

  
  const countQuery = `SELECT
            count(u.id) as 'totalCount',u.index_no
        FROM
            users u
        LEFT JOIN (
            SELECT
                u.id,
                SUM(COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) AS total_score,
                RANK() OVER (ORDER BY SUM( COALESCE(s.knowledge, 0) + COALESCE(s.creativity, 0) + COALESCE(s.problem_solving, 0) + COALESCE(s.communication, 0) +COALESCE(s.descision_making, 0)+COALESCE(s.team_work, 0)+COALESCE(s.leadership, 0)) DESC) AS 'ranks'
            FROM
                users u
            JOIN
                skills s ON u.id = s.user_id
            GROUP BY
                u.id
        ) AS user_scores ON user_scores.id = u.id
        where status like 'active' and role_id = 1 and index_no like '%${search}%'`;

  try {
    const results = await sequelize.query(query, {
      type: QueryTypes.SELECT,
    });

    const countResult = await sequelize.query(countQuery, {
      type: QueryTypes.SELECT,
    });
    return res.status(200).json({
      results: {
        data: results,
        count: countResult[0]?.totalCount,
      },
    });
  } catch (error) {
    console.log(error);
    return res.status(500).send('error while fetching students details');
  }
}
