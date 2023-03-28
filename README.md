<div align="center">
  <img src="https://github.com/RainelDias88/packagePYPI/blob/6a9709f6baaa6ce0051cd0e37ac8a35737fb923d/file/logoraineldataia.png"><br>
</div>

-----------------

## Table of contents

- [Quick start](#quick-start)
- [Status](#status)
- [What is it?](#what-is-it?)
- [Where to get it](#where-to-get-it)
- [Instructions](#instructions)
- [Creator](#creator)

# Disaster Responde: a model for an API that classifies disaster messages.

## Quick start

Several quick start options are available:

- Clone the repo: `git clone https://github.com/RainelDias88/disaster-response`
- [Install Python](https://www.python.org/downloads/)
- [Install Anaconda](https://www.anaconda.com/products/distribution)

## Status

![Languages](https://img.shields.io/github/languages/count/RainelDias88/disaster-response)
![Top Language](https://img.shields.io/github/languages/top/RainelDias88/disaster-response)

## What is it?

In this project I analyzed [Appen](https://appen.com/) disaster data and built a model for an API that classifies disaster messages.

In the 'data' folder there is a dataset containing real messages that were sent during disaster events.

In the 'models' folder there is a 'py' file containing a machine learning pipeline that categorizes the events and 
makes it possible to send the messages to an appropriate disaster relief agency.

In the 'app' folder is a 'py' file containing a web application where an emergency worker can enter a new message and 
get results sorted into various categories. This web application also displays visualizations of the data.

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/RainelDias88/disaster-response

## Instructions:
```sh
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py messages.csv categories.csv disaster_response.db`
    
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py ../data/disaster_response.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
```

## Creator

**Felipe Rainel**

- <https://medium.com/@raineldias88>

- <https://www.linkedin.com/in/felipe-rainel/>

- E-mail: raineldias88@gmail.com
