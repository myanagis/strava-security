import requests
import json
import time
from pandas import json_normalize
import json
import csv

def initial_connect(temp_code:str):
    # Make Strava auth API call with your
    # client_code, client_secret and code
    clientId: int = 96601
    clientSecret: str = "b9b9b274233eff61366001b61b969e065c86498f"

    response = requests.post(
        url='https://www.strava.com/oauth/token',
        data={
            'client_id': clientId,
            'client_secret': clientSecret,
            'code': temp_code,
            'grant_type': 'authorization_code'
        })

    # Save json response as a variable
    strava_tokens = response.json()
    # Save tokens to file
    with open('strava_tokens.json', 'w') as outfile:
        json.dump(strava_tokens, outfile)
    # Open JSON file and print the file contents
    # to check it's worked properly
    with open('strava_tokens.json') as check:
        data = json.load(check)
    print(data)


def test_pull_activities():

    clientId: int = 96601
    clientSecret: str = "b9b9b274233eff61366001b61b969e065c86498f"
    accessToken: str = "e9e45b2e60fddfac011153410e9abf272428aa79"
    refreshToken: str = "0d0c3694b948a27ddc12a116946158d05810822a"
    code: str = '96adbc46a070d05593ec4d20a7fe89e1a5c6f72d'

    #https://www.strava.com/oauth/authorize?client_id=96601&response_type=code&redirect_uri=http://localhost/exchange_token&approval_prompt=force&scope=profile:read_all,activity:read
    #http://localhost/exchange_token?state=&code=916fc592ed529cec4eaa2bd9d280556d829b267a&scope=read,activity:read,profile:read_all


    # code derived from:
    # https://medium.com/swlh/using-python-to-connect-to-stravas-api-and-analyse-your-activities-dummies-guide-5f49727aac86#:~:text=Using%20Python%20to%20Connect%20to%20Strava%E2%80%99s%20API%20and,5%20Automating%20the%20Retrieval%20of%20the%20Tokens%20

    # Save token to file
    with open('strava_tokens.json') as json_file:
        strava_tokens = json.load(json_file)

    # Loop through all activities
    url = "https://www.strava.com/api/v3/activities"
    accessToken = strava_tokens['access_token']

    # Get first page of activities from Strava with all fields
    r = requests.get(url + '?access_token=' + accessToken)
    r = r.json()
    print(r)

    df = json_normalize(r)
    df.to_csv('strava_activites_all_fields.csv')


def connectToStrava():

    clientId: int = 96601
    clientSecret: str = "b9b9b274233eff61366001b61b969e065c86498f"
    accessToken: str = "3b42fae933cab5bb023d37a9e78e29dee5f0ebdd"
    refreshToken: str = "0d0c3694b948a27ddc12a116946158d05810822a"
    code: str = '916fc592ed529cec4eaa2bd9d280556d829b267a'

    #https://www.strava.com/oauth/authorize?client_id=96601&response_type=code&redirect_uri=http://localhost/exchange_token&approval_prompt=force&scope=profile:read_all,activity:read
    #http://localhost/exchange_token?state=&code=916fc592ed529cec4eaa2bd9d280556d829b267a&scope=read,activity:read,profile:read_all


    # code derived from:
    # https://medium.com/swlh/using-python-to-connect-to-stravas-api-and-analyse-your-activities-dummies-guide-5f49727aac86#:~:text=Using%20Python%20to%20Connect%20to%20Strava%E2%80%99s%20API%20and,5%20Automating%20the%20Retrieval%20of%20the%20Tokens%20

    # Save token to file
    with open('strava_tokens.json') as json_file:
        strava_tokens = json.load(json_file)

    # Loop through all activities
    url = "https://www.strava.com/api/v3/activities"
    accessToken = strava_tokens['access_token']

    # Get first page of activities from Strava with all fields
    r = requests.get(url + '?access_token=' + accessToken)
    r = r.json()

    df = json_normalize(r)
    df.to_csv('strava_activites_all_fields.csv')

    #######

    # Save token to file
    with open('strava_tokens.json') as json_file:
        strava_tokens = json.load(json_file)

    # Make strava auth API call with current refresh token
    if strava_tokens['expires_at'] < time.time():
        response = requests.post(url = 'https://www.strava.com/oauth/token',
                             data = {
                                 'client_id': clientId,
                                 'client_secret': clientSecret,
                                 'grant_type': 'refresh_token',
                                 'refresh_token': strava_tokens['refresh_token']
                             }
                             )

    # Save json response as a variable
    new_strava_tokens = response.json()

    # Open JSON file and print the file contents to check it's worked properly
    with open('strava_tokens.json', 'w') as outfile:
        json.dump(new_strava_tokens, outfile)

    # Use new strava tokens
    strava_tokens = new_strava_tokens

    # Open new JSON file
    with open('strava_tokens.json') as check:
        data = json.load(check)
    print(data)