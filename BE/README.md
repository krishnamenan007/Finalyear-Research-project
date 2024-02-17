# MAN POWER API - node, express with mysql

Node.js + MySQL API for User Management, Authentication and Registration

# How to register a new user with Postman / insomnia
To register a new user with the api follow these steps:

- Register (POST) - http://localhost:4000/admin/register
```
{
    "firstName": "shan",
    "lastName": "jathu",
    "email": "shan@gmail.com",
    "password": "my-super-secret-password"
}
```

- Login (POST) - http://localhost:4000/admin/login

```
{
    "email": "shan@gmail.com",
    "password": "my-super-secret-password"
}
```

- How to make an authenticated request to retrieve all users \
 (GET) http://localhost:4000/users \
Select the "Authorization" tab below the URL field, change the type to "Bearer Token" in the type dropdown selector, and paste the JWT token from the previous authenticate step into the "Token" field.

- Get user by Id (GET) - http://localhost:4000/admin/1

- update user by Id (PUT) - http://localhost:4000/admin/1 
```
{
    "firstName": "shan",
    "lastName": "jathu",
    "email": "shanjathu@gmail.com",
    "password": "123456"
}
```

- RESET PASSWORD
```
api 1 - method (POST)
http://localhost:4000/admin/sendOTP 
body  => { "email" : "shanaa@gmail.com" }

api 2 - method (POST) 
http://localhost:4000/admin/checkOTP 
body => { "email" : "shanaa@gmail.com", "resetotp": "66975" }

api 3 - method (PUT) 
http://localhost:4000/admin/shanaa@gmail.com
body => { "password": "123456" }
```

# API URLs
Paths = **[ users, subjects, grades, types, terms ]**

```
View all (GET)   - http://localhost:4000/{Paths}
View by Id (GET) - http://localhost:4000/{Paths}/5
Create (POST)    - http://localhost:4000/{Paths}
Update (PUT)     - http://localhost:4000/{Paths}
Delete (DELETE)  - http://localhost:4000/{Paths}/7
```

Have a good day!
- Developer - SHANJATHURSHAN.
# manpower-api
# mfc-dca-api-nodejs-april-11-2022
# mfc-dca-api-nodejs-may-02-2022
# mfc-dca-api-nodejs-may-02-2022
# mfc-dca-api-nodejs-may-02-2022
