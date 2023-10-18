Create a Google Cloud Project:

Go to the Google Cloud Console.
Create a new project or select an existing one.
Enable Google Sheets API:

In the Cloud Console, go to the "APIs & Services" -> "Library."
Search for "Google Sheets API" and enable it for your project.
Create Service Account Credentials:

In the Cloud Console, go to the "APIs & Services" -> "Credentials."
Click on "Create credentials" and select "Service Account Key."
Fill in the details for the service account, such as name and role (e.g., Editor).
Choose "JSON" as the key type.
Click "Create" to create the service account and download the JSON key file.
Share the Google Sheet with the Service Account:

Open the Google Sheet you provided (https://docs.google.com/spreadsheets/d/14bJsdFKWzQeKb0Eiv57spucvGFpBejID7Kje6LwsP8w/edit?pli=1#gid=0).
Click on the "Share" button in the top right.
Share the sheet with the email address associated with the service account. You can find this in the downloaded JSON key file.
Use the Service Account Credentials in Your Code:

Place the downloaded JSON key file in a location accessible from your code.
Use the path to the JSON key file in your code: