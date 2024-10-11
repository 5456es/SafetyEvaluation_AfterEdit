import requests
import json
from bs4 import BeautifulSoup


url='https://en.wikipedia.org/wiki/2024'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

### goto <div class="mw-content-container">
content = soup.find('div', class_='mw-content-container')


### iterate li in  <div class="mw-heading mw-heading3">
        #    <h3 id="Month">
        #     Month
        #    </h3>
        #   </div>
        #   <ul>


year_2024_news_raw = []
for heading in content.find_all('div', class_='mw-heading mw-heading3'):
    for li in heading.find_next('ul').find_all('li'):
        print(li.text)
        li_list=li.text.split('\n')
        year_2024_news_raw.extend(li_list)

months=['January','February','March','April','May','June','July','August','September','October','November','December']

def dayline(line):
    ### format like   "January 14",
    for month in months:
        if len(line)<20 and month in line:
            return line
    return None
current_day = None

def start_with_month(line):
    for month in months:
        if line.startswith(month):
            return True
    return False

for i in range(len(year_2024_news_raw)):
    result =dayline(year_2024_news_raw[i])
    if result:
        current_day = result
    elif(start_with_month(year_2024_news_raw[i])):
        continue
    else:
        year_2024_news_raw[i] = current_day + ' ' + year_2024_news_raw[i]


total=len(year_2024_news_raw)-1
for i in range(len(year_2024_news_raw)):
    if dayline(year_2024_news_raw[total-i]):
        year_2024_news_raw.pop(total-i)
    

###save as csv
import pandas as pd
df=pd.DataFrame(year_2024_news_raw,columns=['news']
)   
df.to_csv('2024_news_raw.csv',index=False)
        
