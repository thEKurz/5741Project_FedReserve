import pandas as pd
import numpy as np
import re
import flair

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
month_classifier = {}
i = 1
for m in months:
    month_classifier[m] = i
    i += 1
year_reg = "[0-9]{4}"

df_text = pd.DataFrame(columns = ["Year", "Month", "Sentiment"])

num_files = 118
for i in range(1, (num_files + 1)):
    fname = "fed_press_data\\" + str(i) + ".txt"
    print(fname)
    with open(fname, encoding = "utf8") as fhand:
        date_m = 0
        sentence = ""
        for line in fhand:
            line=line.rstrip() #This removes the new line character at the end of the line
            line += ' '
            if date_m == 0:
                for m in months:
                    if m in line:
                        date_m = month_classifier[m]
                        date_y = re.findall(year_reg, line)[0]
                        date_y = int(date_y)
            
            # Put all lines into one string for analysis
            sentence += line
        
        s = flair.data.Sentence(sentence)
        flair_sentiment.predict(s)
        total_sentiment = s.labels[0].to_dict()
        if total_sentiment['value'] == "NEGATIVE" :
            senti = total_sentiment['confidence'] * (-1)
        else: 
            senti = total_sentiment['confidence']
        
        # The sentimental array has values in order "negative", "neutral", "positive"
    
    df_text.loc[i-1] = [date_y, date_m, senti]

df_text['Date'] = pd.to_datetime(df_text[['Year', 'Month']].assign(DAY=1))
df_text = df_text.set_index('Date')
df_text = df_text.drop(["Year", "Month"], axis = 1)

df_text.to_csv("text_sent_flair.csv", index=True)
