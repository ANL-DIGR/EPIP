import matplotlib
matplotlib.use('Agg')

import act
import glob
import matplotlib.pyplot as plt
import json
import sys

#Read in ARM Live Data Webservice Token and Username
with open('./token.json') as f:
    data = json.load(f)
username = data['username']
token = data['token']

#Specify datastream and date range for KAZR data
site = 'sgp'
startdate = '2017-01-01'
enddate = '2019-12-31'

sdate = ''.join(startdate.split('-'))
edate = ''.join(enddate.split('-'))

datastream = site+'vdisdropsC1.b1'
result = act.discovery.download_data(username, token, datastream, startdate, enddate)
