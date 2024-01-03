# Server Back-End for the Quizz Application

## Table of Contents

1. [Overview](#overview)
   - [Working](#working)
   
2. [How to Use](#how-to-use)
   - [Run Dev](#run-dev)
   - [Run Prod](#run-dev)
   - [Connection](#connection)
3. [Project Big Picture](#project-big-picture)

## Overview

This part of the project contain the current back-end that will serve the front end for the web application. Its contain basics thing.  
The whole server is containerizer into a docker service.

### Working

This work as all usual Flask application.

It used PostgresSQL at database, and Redis to manage user session. For the local part all these service are part of the network of the docker, but in production, make sure to use another service like AWS to allow scaling.  
All the configuration are in the template folder. You'll need to fill these before event start the application.   
Make sure beforehand to have created an S3 user and fill all the credentials.

## How to Use

### Run dev

As said before, please make sure that your env file (at least the dev one) are filled with your personnal information.  
Then run:
```bash
cd ..
docker-compose -f docker-compose.dev.yaml up --build -d
```
It will start all the container and allow the application to be live.   
For the dev part, admin user will be created, please consult the `manage.py` and the `entrypoint.dev.sh` file to see more details

### Run prod

As said before, please make sure that your env file (at least the prod one) are filled with your personnal information.
Then run
```bash
cd ..
docker-compose -f docker-compose.prod.yaml up --build -d
```
It will start all the container and allow the application to be live.   
Please consult the `manage.py` and the `entrypoint.dev.sh` file to see more details
Please make sure if using multiple servers to use only one Postgre SQL service and one REDIS service. They are still in the docker compose but it is not recommanded to use them, feel free to remove them


### Connection
To perfom query, please hit the 5001 port within localhost. 

## Project Big Picture

Without getting into details, the back-end are currently support:

- Register (Server Side)
- Login (Server Side)
- Logout (Server Side)
- Get user infos
- Manager User infos
- Delete User
- Create quizz (Only for admin)
- Sumbit Quizz anwsers
- Access to Quizz
