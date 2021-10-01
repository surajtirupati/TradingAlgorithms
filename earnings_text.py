from bs4 import BeautifulSoup
from datetime import datetime
from twilio.rest import Client
import requests
import pandas as pd

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'close',
    'DNT': '1', # Do Not Track Request Header
    'Pragma': 'no-cache',
    'Referrer': 'https://google.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}

# initialising empty dataframe which will hold all tickers
df = pd.DataFrame()

for i in range(1, 5):
    # the URL which updates on each loop iteration
    url = "https://seekingalpha.com/earnings/earnings-calendar/{}".format(i)

    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url, headers=headers).text

    # get the HTML content
    soup = BeautifulSoup(html_content, "lxml")

    # obtaining tables
    table = soup.find_all('table')

    # indexing the relevant tables to obtain data of interest
    data = pd.read_html(str(table))[0].set_index("Release Date").iloc[:, :-1]

    # appending the data on the new page to the dataframe
    df = df.append(data)

# converting the dataframes index to datetime
df.index = pd.to_datetime(df.index)

# obtaining today's date
today = datetime.today().strftime('%Y-%m-%d')

# obtaining companies reporting today
todays_companys = df[df.index == today]

# initialising a list to hold strings of each line of text message
info = []

# appending each line of text message into info[]
for i in range(len(todays_companys)):

    # indexing rows of the dataframe
    todays_companys.iloc[i, :]

    # formatting the string message
    string = "{} ({}) is reporting today. The release time is {}.".format(todays_companys.iloc[i, :]["Name"],
                                                                          todays_companys.iloc[i, :]["Symbol"],
                                                                          todays_companys.iloc[i, :]["Release Time"])
    # appending each line of text into info
    info.append(string)

# joining info into one long string
msg = "\n".join(info)

# Twilio Account SID and Auth Token
client = Client("ACe737e611e4524cdf0ed7feb19d4ef564", "044ca567300180705e108e76715dd60e")

# Sending text message to my mobile
client.messages.create(to="+447719104171",
                       from_="+447897033826",
                       body=msg)

return