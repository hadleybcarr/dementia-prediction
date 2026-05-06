import requests
import json
from urllib.parse import urlencode
import webbrowser
import os
from dotenv import load_dotenv


url_1 = "https://cloud.ouraring.com/oauth/authorize?client_id=b04c3fa6-07e0-4b29-9504-4d51d324e99d&redirect_uri=http%3A%2F%2Flocalhost%3A5173%2F&response_type=code&scope=daily+heartrate+personal"
#url_2 = "https://cloud.ouraring.com/oauth/authorize?client_id=b04c3fa6-07e0-4b29-9504-4d51d324e99d&redirect_uri=http%3A%2F%2Flocalhost%3A5173&response_type=code&scope=email+personal+daily+heartrate+tag+workout+session+spo2+ring_configuration+stress+heart_health
# Your OAuth2 application credentials
load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI")

# Step 1: Direct user to authorization page
auth_params = {
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
   "scope": "personal daily heartrate spo2 sleep heart_health"
}
auth_url = f"https://cloud.ouraring.com/oauth/authorize?{urlencode(auth_params)}"
print(f"Please visit this URL to authorize: {auth_url}")
webbrowser.open(auth_url)

# Step 2: Exchange authorization code for access token
# After user authorizes, they'll be redirected to your redirect URI with a code parameter
auth_code = input("Enter the authorization code from the redirect URL: ")

token_url = "https://api.ouraring.com/oauth/token"
token_data = {
    "grant_type": "authorization_code",
    "code": auth_code,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI
}
response = requests.post(token_url, data=token_data)
tokens = response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]

# Step 3: Use the access token to make API calls
headers = {"Authorization": f"Bearer {access_token}"}
heart_rate = requests.get(
    "https://api.ouraring.com/v2/usercollection/heartrate",
    headers=headers,
)
print(json.dumps(heart_rate.json(), indent=2))

blood_oxygen = requests.get(
    "https://api.ouraring.com/v2/usercollection/daily_spo2",
    headers = headers,
    params = {}
)

print(json.dumps(blood_oxygen.json(), indent=2))

vo2_max = requests.get(
    "https://api.ouraring.com/v2/usercollection/vO2_max",
    headers = headers,
    params = {}
)

print(json.dumps(vo2_max.json(), indent=2))

sleep = requests.get(
    "https://api.ouraring.com/v2/usercollection/sleep",
    headers=headers,
    params={"start_date": "2023-01-01", "end_date": "2023-01-07"},
)
print(json.dumps(sleep.json(), indent=2))



# Step 4: Refresh the token when it expires
def refresh_access_token(refresh_token):
    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    response = requests.post(token_url, data=token_data)
    new_tokens = response.json()
    return new_tokens["access_token"], new_tokens["refresh_token"]