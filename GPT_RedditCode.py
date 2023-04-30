import praw
import pandas as pd
import re
import string
import my_credentials
# Create a Reddit instance using PRAW
#Different queries output different parameters to use here, will need to check on that
#reddit = praw.Reddit(client_id='your_client_id',
#                     client_secret='your_client_secret',
#                     user_agent='your_user_agent')
reddit = praw.Reddit(client_id=my_credentials.CLIENT_ID, client_secret=my_credentials.CLIENT_SECRET, user_agent="PSYC593_script:v1 (by /u/Champaign__Supernova)", password=my_credentials.PASSWORD, username=my_credentials.USERNAME)
# Define the subreddits you want to scrape
DEFAULT1 = ["explainlikeimfive","offmychest","Showerthoughts","mildlyinteresting","mildlyinfuriating","LifeProTips","UIUC","OldSchoolCool","DIY","movies","worldnews","politics"]
DEFAULT5 = ["TwoXChromosomes","changemyview","unpopularopinion","OutOfTheLoop","YouShouldKnow","IAmA","memes","NoStupidQuestions","relationship_advice","gaming","food","science","amitheasshole"]
DEFAULT4 = ["interestingasfuck","whitepeopletwitter","blackpeopletwitter","dankmemes","wholesomememes","idiotsincars"]
DEFAULT2 = ["askscience","Art","books","nottheonion","sports","SkincareAddiction","UIUC"]
DEFAULT3 = ["Music","pics","videos","MaliciousCompliance","todayilearned","aww","Funny","AdviceAnimals"]
subreddits = DEFAULT1+DEFAULT5#+DEFAULT4+DEFAULT2+DEFAULT3

#TIFU removed due to some weird new rule impacting comments?
# Define the columns of the CSV file
columns = ['subreddit', 'text']

# Create an empty Pandas dataframe to store the scraped data
df = pd.DataFrame(columns=columns)

#for printint outputs of progress
LIM = int(input("how many posts from each subreddit would you like to scrape? "))
counter2 = 0

# Loop through each subreddit and scrape the text from posts and comments
for subreddit_name in subreddits:
    counter = 0
    counter2 += 1
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.hot(limit=LIM):
        counter +=1
        print((counter/LIM)*100, "% of the way there for subreddit ", counter2," of ",len(subreddits))
        # Append the post text to the dataframe

        df = df.append({'subreddit': subreddit_name,
                        'text': submission.title + " " + submission.selftext}, ignore_index=True)
        # Loop through the comments in the post and append them to the dataframe
        #submission.comments.replace_more(limit=None)
        #for comment in submission.comments.list():
        #s    df = df.append({'subreddit': subreddit_name, 'text': comment.body}, ignore_index=True)

STAHP = [",",".","...","?","!",")","(","“","”",'"',";","^","<",">","$","[","]","{","}","_","*","~"]
def clean_up(text):
    text = ",".join([i for i in text if i not in STAHP]) #super hacky way to get rid of punctuation lol
    text = re.sub(r',','',text) #this is the hack
    text = re.sub("#[A-Za-z0-9]+", "", text) #removing hashtags
    text = re.sub("/"," ",text)
    text = re.sub("’", " '",text)
    text = re.sub("'"," '", text)
    text = re.sub("[/*]{2,}","",text)
    text = re.sub("'","'",text)
    text = re.sub("\s+"," ", text)
    text = text.lower()
    return text

def remove_unprintable_chars(text_as_string):
    printables = list(string.printable)
    filtered_printable = "".join(filter(lambda x: x in printables, text_as_string))
    return filtered_printable

def remove_tags_and_links(text):
    #Basically a funciton for removing the unique text things for a given site
    cleaned = re.sub("#{3,}","", text)
    cleaned = re.sub("[\W+]http(s)*\S+", " ", cleaned) #removing urls and urls enclosed in parentheses (linking format for text)
    cleaned = re.sub("http(s)*\S+", " ", cleaned) #removing urls and urls enclosed in parentheses (linking format for text)
    cleaned = re.sub("@\S+", " ", cleaned) #removing usernames (maybe want to add back in later if you look at behavior when atting versus hashtags/general posting)
    cleaned = re.sub("#\S+", " ", cleaned) #removing hashtags
    cleaned = re.sub("(?<!\w)r/\S+", "SUBREDDITNAME", cleaned) #removing hashtags #PUT NETGATIVE LOOKBACK FOR ANYTHING OTHER THAN JUST r or u
    cleaned = re.sub("(?<!\w)u/\S+", "REDDITUSERNAME", cleaned) #removing hashtags
    cleaned = re.sub("\[deleted\]", " ", cleaned) #removing deleted posts because they are deleted
    cleaned = re.sub("\[removed\]", " ", cleaned) #removing removed posts because they are removed
    cleaned = re.sub("!\[gif\]\S+", "gif", cleaned) #gif links are weird
    cleaned = re.sub("\(/message\S+\)", " ", cleaned)
    cleaned = re.sub("\<a href =.* \>", " ", cleaned) #lol html
    cleaned = re.sub("\<a.* \>", " ", cleaned)
    cleaned = clean_up(cleaned)
    cleaned = remove_unprintable_chars(cleaned)
    return cleaned

df['cleaned'] = df['text'].apply(lambda x: pd.Series(remove_tags_and_links(x)))
print("program ran successfully, now saving file...")
# Save the dataframe to a CSV file
df.to_csv('reddit_data4.csv', index=False)
