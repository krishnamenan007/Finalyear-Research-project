const express = require('express');
const app = express();
const router = express.Router();
const verifyToken = require('../_middleware/authorize');
const db = require('../_helpers/dbModel');
const { Sequelize, QueryTypes } = require('sequelize');
const config = require('../_helpers/config.json');
const { func } = require('joi');

router.get('/', verifyToken, getProfileDetails);
// Add this line to add a new route for updating approved status
router.post('/updateApproved', verifyToken, updateApprovedStatus);
router.post('/updateRejected', verifyToken, updateRejectStatus);
module.exports = router;


async function getProfileDetails(req, res) {
    const { page = 1, limit = 10, search } = req.query; // providing default values

    const offset = (page - 1) * limit;

    const { host, port, user, password, database } = config.database;

    const sequelize = new Sequelize(database, user, password, {
        dialect: 'mysql',
    });


    const query = `
    SELECT 
    'selflearning' as type,  
    u.index_no,
    u.first_name,
    sl.course_level AS level,
    f1.file_data,
    u.id AS user_id,
    sl.id AS item_id  -- Changed alias to item_id
FROM 
    users u
LEFT JOIN 
    selflearnings sl ON u.id = sl.user_id
LEFT JOIN 
    files f1 ON sl.file_id = f1.id
WHERE 
    u.index_no IS NOT NULL AND 
    sl.course_level IS NOT NULL AND 
    f1.file_data IS NOT NULL AND
    sl.approved = FALSE AND
    sl.rejected = FALSE

UNION ALL

SELECT 
    'sports' as type,  
    u.index_no,
    u.first_name,
    s.level,
    f2.file_data,
    u.id AS user_id,
    s.id AS item_id  -- Changed alias to item_id
FROM 
    users u
LEFT JOIN 
    sports s ON u.id = s.user_id
LEFT JOIN 
    files f2 ON s.file_id = f2.id
WHERE 
    u.index_no IS NOT NULL AND 
    s.level IS NOT NULL AND 
    f2.file_data IS NOT NULL AND
    s.approved = FALSE AND
    s.rejected = FALSE;
`;

    try {
        const results = await sequelize.query(query, {
            type: QueryTypes.SELECT,
        });

        return res.status(200).json({
            results: {
                data: results
            },
        });
    } catch (error) {
        console.log(error);
        return res.status(500).send('Error');
    }
}




// Helper function to calculate skills
function calculateSkills(level, position, cardinality, existingSkill) {
    let teamwork = 0, leadership = 0, descision_making = 0;

    if (level === 'university') {
        if (position === '1') {
            if (cardinality === 'leader') {
                descision_making = 3;
                leadership = 2;
                teamwork = 2;
            }
            if (cardinality === 'member') {
                descision_making = 3;
                leadership = 0;
                teamwork = 4;
            }
            if (cardinality === 'individual') {
                descision_making = 7;
                leadership = 0;
                teamwork = 0;
            }
        }
        if (position === '2') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 2;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 3;
                leadership = 0;
                descision_making = 3;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 6;
            }
        }

        if (position === '3') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 1;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 3;
                leadership = 0;
                descision_making = 2;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 5;
            }
        }

        if (position === 'part') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 0.5;
                descision_making = 0.5;
            }
            if (cardinality === 'member') {
                teamwork = 1;
                leadership = 0;
                descision_making = 1;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 2;
            }
        }
    } else if (level === 'national') {
        if (position === '1') {
            if (cardinality === 'leader') {
                teamwork = 2;
                leadership = 3;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 4;
                leadership = 0;
                descision_making = 4;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 8;
            }
        }

        if (position === '2') {
            if (cardinality === 'leader') {
                teamwork = 2;
                leadership = 2;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 4;
                leadership = 0;
                descision_making = 3;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 7;
            }
        }

        if (position === '3') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 2;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 3;
                leadership = 0;
                descision_making = 3;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 6;
            }
        }

        if (position === 'part') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 0.5;
                descision_making = 0.5;
            }
            if (cardinality === 'member') {
                teamwork = 1;
                leadership = 0;
                descision_making = 1;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 2;
            }
        }
    } else if (level === 'international') {
        if (position === '1') {
            if (cardinality === 'leader') {
                teamwork = 2;
                leadership = 3;
                descision_making = 5;
            }
            if (cardinality === 'member') {
                teamwork = 5;
                leadership = 0;
                descision_making = 5;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 10;
            }
        }
        if (position === '2') {
            if (cardinality === 'leader') {
                teamwork = 2;
                leadership = 2;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 4;
                leadership = 0;
                descision_making = 4;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 8;
            }
        }

        if (position === '3') {
            if (cardinality === 'leader') {
                teamwork = 2;
                leadership = 2;
                descision_making = 3;
            }
            if (cardinality === 'member') {
                teamwork = 4;
                leadership = 0;
                descision_making = 3;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 7;
            }
        }

        if (position === 'part') {
            if (cardinality === 'leader') {
                teamwork = 1;
                leadership = 0.5;
                descision_making = 0.5;
            }
            if (cardinality === 'member') {
                teamwork = 1;
                leadership = 0;
                descision_making = 1;
            }
            if (cardinality === 'individual') {
                teamwork = 0;
                leadership = 0;
                descision_making = 2;
            }
        }
    }

    if (existingSkill) {
        teamwork += existingSkill.teamwork;
        descision_making += existingSkill.descision_making;
        leadership += existingSkill.leadership;
    }

    return { teamwork, leadership, descision_making };
}


async function updateRejectStatus(req, res){
    const { type, item_id} = req.body;

    if (!type || !item_id) {
        return res.status(400).send('Type and item_id are required');
    }

    try {

        if(type === 'selflearning') {
            await db.SelfLearning.update({ rejected: true}, { where: {id: item_id}});
            updateRecord = await db.SelfLearning.findOne({ where: {id: item_id}});
        } else if (type === 'sports') {
            await db.Sport.update({ rejected: true }, { where: { id: item_id } });
            updatedRecord = await db.Sport.findOne({ where: { id: item_id } });
        } else {
            return res.status(400).send('Invalid type');
        }

        
        return res.status(200).send('Approved status and skills updated successfully');
    }catch (error) {
        console.log(error);
        return res.status(500).send('Error updating approved status and skills');
    }
}


async function updateApprovedStatus(req, res) {
    const { type, item_id } = req.body;

    if (!type || !item_id) {
        return res.status(400).send('Type and item_id are required');
    }

    try {
        let updatedRecord;
        if (type === 'selflearning') {
            await db.SelfLearning.update({ approved: true }, { where: { id: item_id } });
            updatedRecord = await db.SelfLearning.findOne({ where: { id: item_id } });
        } else if (type === 'sports') {
            await db.Sport.update({ approved: true }, { where: { id: item_id } });
            updatedRecord = await db.Sport.findOne({ where: { id: item_id } });
        } else {
            return res.status(400).send('Invalid type');
        }

        if (type === 'sports' && updatedRecord) {
            const { level, position, cardinality, user_id } = updatedRecord;

            // Fetch existing skills
            const existingSkill = await db.Skill.findOne({ where: { user_id } });

            // Call the calculateSportsSkills function
            // In updateApprovedStatus
            const { teamwork, leadership, descision_making } = calculateSkills(level, position, cardinality, existingSkill);

            if (existingSkill) {
                await db.Skill.update(
                    {
                        leadership,
                        descision_making,
                        team_work: teamwork,
                    },
                    { where: { user_id } }
                );
            } else {
                await db.Skill.create({
                    leadership,
                    descision_making,
                    team_work: teamwork,
                    user_id,
                });
            }
        }
        // Existing self-learning skill logic here
        if (type === 'selflearning' && updatedRecord) {
            const { course_level, user_id } = updatedRecord;
            let knowledgeSkill = 0, psSkill = 0, creativity = 0;

            if (course_level === 'beginner') {
                knowledgeSkill = 5;
                psSkill = 5;
                creativity = 5;
            } else if (course_level === 'intermediate') {
                knowledgeSkill = 10;
                psSkill = 5;
            } else if (course_level === 'expert') {
                knowledgeSkill = 10;
                psSkill = 10;
                creativity = 5;
            }

            // Fetch existing skills
            const existingSkill = await db.Skill.findOne({ where: { user_id } });

            if (existingSkill) {
                knowledgeSkill += existingSkill.knowledge;
                psSkill += existingSkill.problem_solving;
                creativity += existingSkill.creativity;

                await db.Skill.update(
                    {
                        creativity,
                        problem_solving: psSkill,
                        knowledge: knowledgeSkill,
                    },
                    { where: { user_id } }
                );
            } else {
                await db.Skill.create({
                    creativity,
                    problem_solving: psSkill,
                    knowledge: knowledgeSkill,
                    user_id,
                });
            }
        }
        console.log("ewee:",pendingItems);
        return res.status(200).send('Approved status and skills updated successfully');
    } catch (error) {
        console.log(error);
        return res.status(500).send('Error updating approved status and skills');
    }
}
app.get('/api/pendingItems', async (req, res) => {
    try {
        const pendingSelfLearningItems = await db.SelfLearning.findAll({ where: { approved: false, rejected: false } });
        const pendingSportsItems = await db.Sport.findAll({ where: { approved: false, rejected: false } });
        // Add more logic to fetch other types of pending items if needed
         
        
        const pendingItems = {
            selfLearning: pendingSelfLearningItems,
            sports: pendingSportsItems,
            
            // Add more properties for other types of items if needed
        };
        
        res.status(200).json({ status: 'success', data: pendingItems });
    } catch (error) {
        console.error('Error fetching pending items:', error);
        res.status(500).json({ status: 'error', message: 'Internal server error' });
    }
});
