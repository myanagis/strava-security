import stravalib

# https://pythonhosted.org/stravalib/

def connect():

    clientId: int = 96601
    clientSecret: str = "b9b9b274233eff61366001b61b969e065c86498f"
    accessToken: str = "3b42fae933cab5bb023d37a9e78e29dee5f0ebdd"
    refreshToken: str = "0d0c3694b948a27ddc12a116946158d05810822a"
    code: str = '916fc592ed529cec4eaa2bd9d280556d829b267a'

    authorization_uri = "http://127.0.0.1:5000/authorization"

    client = stravalib.Client()
    url = client.authorization_url(client_id=clientId,
                                   redirect_uri=authorization_uri)

    print(url)

