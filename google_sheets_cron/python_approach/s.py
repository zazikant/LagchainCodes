#!pip install gspread
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('s.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1jL777b7oVBbVjgmbOZcUbOFxNy6d-g6ZfSrvkeHmw0k/edit#gid=0').sheet1

# Initialize a variable to store the last checked values
last_checked_data = None

while True:
    # Get the current data in the sheet
    current_data = sheet.get_all_values()

    # Check if the data has changed since the last check
    if current_data != last_checked_data and current_data:
        # Process the data to find unique values based on email
        unique_data = []
        last_instance_dict = {}

        for row in current_data[1:]:  # Skip header row
            email = row[0].lower() if row else ""
            if email:
                last_instance_dict[email] = row

        # Append the header row if there are rows in the sheet
        if current_data:
            unique_data.append(current_data[0])

        # Append the last instance for each unique email
        unique_data.extend(list(last_instance_dict.values()))

        # Update only the necessary rows
        sheet.clear()
        sheet.append_rows(unique_data)

        # Update the last checked data
        last_checked_data = current_data

        # print('Sheet updated! Unique data:', unique_data)

    # Wait for some time before checking again (e.g., every 5 seconds)
    time.sleep(1)
