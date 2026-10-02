import os
import requests
import json
import time
import logging

# --- Credentials come from the environment (never hardcode them) ---
#   export STRAVA_CLIENT_ID=...  STRAVA_CLIENT_SECRET=...  STRAVA_REFRESH_TOKEN=...
CLIENT_ID = os.environ.get("STRAVA_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("STRAVA_REFRESH_TOKEN", "")

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()


def refresh_access_token():
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN
    }
    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise RuntimeError(
            "Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET and STRAVA_REFRESH_TOKEN environment variables."
        )
    logger.info("Refreshing access token...")
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        logger.error(f"Token refresh failed: {response.status_code} {response.text}")
        raise Exception("Failed to refresh token")
    data = response.json()
    if "access_token" not in data:
        logger.error(f"Token response missing 'access_token': {data}")
        raise Exception("No access token in response")
    logger.info("Access token refreshed successfully.")
    return data["access_token"]

def safe_api_request(url, headers, retries=3):
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 429:  # rate limited
            wait_time = 60 * (attempt + 1)
            logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue
        if response.status_code >= 500:
            wait_time = 5 * (attempt + 1)
            logger.warning(f"Server error {response.status_code}. Retry in {wait_time}s...")
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            logger.error(f"Failed request {response.status_code}: {response.text}")
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error("Invalid JSON response. Skipping this page.")
            return None
    logger.error("Max retries exceeded for API request.")
    return None

def get_activities(access_token):
    activities = []
    page = 1
    per_page = 50

    logger.info("Starting activity download...")

    while True:
        logger.info(f"Fetching page {page}...")
        url = f"https://www.strava.com/api/v3/athlete/activities?page={page}&per_page={per_page}"
        headers = {"Authorization": f"Bearer {access_token}"}

        data = safe_api_request(url, headers)
        if data is None:
            logger.warning(f"Skipping page {page} due to API error.")
            page += 1
            continue

        if not data:
            logger.info("No more activities found.")
            break

        logger.info(f"Downloaded {len(data)} activities from page {page}.")
        activities.extend(data)
        page += 1
        time.sleep(0.2)

    logger.info(f"Total activities downloaded: {len(activities)}")
    return activities

if __name__ == "__main__":
    try:
        token = refresh_access_token()
        all_activities = get_activities(token)

        output_file = "strava_activities.json"
        with open(output_file, "w") as f:
            json.dump(all_activities, f, indent=4)
        logger.info(f"Activities saved to {output_file}")
    except Exception as e:
        logger.exception("Error during execution")