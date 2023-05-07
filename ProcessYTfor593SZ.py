from youtube_transcript_api import YouTubeTranscriptApi
import re
import csv
import pandas as pd
import os
import numpy as np
import time
import datetime
from datetime import datetime
import sys
import spacy
import lxml
import urllib
import string
from lxml import etree
from pytube import YouTube
import youtube_transcript_api
SPACY_MODEL = spacy.load("en_core_web_lg")
SPACY_MODEL.max_length = 100000000
NOW = datetime.now()
LAST = str(sys.argv[1]) #Add "Yes" when you're on the last run of this to combine all the dataframes into one smaller one. You'll have to move them manually ahead of time. Otherwise type no
TIMESTRING = NOW.strftime("%m-%d-%Y_%H_%M")
NOW = datetime.now()
TIMESTRING = NOW.strftime("%m-%d-%Y_%H_%M")
END = [".",'!',"?",",",":",";","-",":","/", "(",")","~"]
PUNCT = [".","?","!",")","(",";","[","]","{","}","^","~","*","=","~","|","…",'"',"$","#","@","<",">","/","'","_","-"]
FILE_NAME_LIST = ['entertainment_video_links.csv','beauty_and_fashion_video_links.csv','food_cooking_and_health_links.csv','videogame_links.csv','educational_video_links.csv','insider_links.csv']#,'sports_video_links.csv']#]

pattern = r'(?:\?v=|&v=|/)([-\w]+)(?:&|$)'


def gather_file_names():
    ''' gather all your individual input files that have the links to the youtube videos, since they are broken up
    Returns a list of files that you will read in next '''
    INPUT_FOR_DFS = []
    DONE = False
    while DONE == False:
        file_name = str(input("please enter a file name and be sure to not add quotes but do include the extension. type DONE when done: "))
        if file_name == "DONE" or file_name == "Done" or file_name == "done":
            DONE = True
        else:
            INPUT_FOR_DFS.append(file_name)
    return INPUT_FOR_DFS

def read_csvs_make_DFs_List(FILE_NAME_LIST):
    ''' Reads in CSV files with the links and various attribute info about the videos, such as author and labeled category. Use stored output from last function
    Returns a list of pandas dataframes (individual) that will be concatenated in the next function '''
    MASTER_DF_LIST = []
    for FILE_NAME in FILE_NAME_LIST:
        csvdf =pd.read_csv(FILE_NAME, sep = ",", encoding="utf-8")
        MASTER_DF_LIST.append(csvdf)
        print(MASTER_DF_LIST)
    return MASTER_DF_LIST

def concat_all_dfs(DF_LIST):
    ''' Takes that list of data frames that each came from a separate csv file, and puts them into 1 dataframe.
    They all have the same column names, and there's no reason not to have one dataframe '''
    df_to_start = DF_LIST[0] #start with a df so you have something to append to
    counter = 0 #use while loop so make counter
    while counter < len(DF_LIST)-1: #while you still have dfs to go through
        df_to_start = pd.concat([df_to_start, DF_LIST[counter+1]], axis=0) #add them together and re-write
        counter += 1
    return df_to_start

def remove_unprintable_chars(text_as_string):
    printables = list(string.printable)
    filtered_printable = "".join(filter(lambda x: x in printables, text_as_string))
    return filtered_printable

def clean_titles(title):
    clean_list = ["(", ")","[","]","{","}","|", ",", ":",";", "-","~","*","&","^","$","%","#","@","!","+","=",".","?","/","'",'"',"-"]
    for thing in clean_list:
        title.replace(thing, " ")
    try:
        title = re.sub("- YouTube","", title)
    except:
        try:
            title = re.sub("Youtube", "", title)
        except:
            pass
    title = re.sub("\s{2,}"," ", title)
    title = re.sub("\s", "_", title)
    title_string = ",".join([i for i in title if i not in clean_list])
    title_string = re.sub(r',','',title_string)
    title_string = remove_unprintable_chars(title_string)
    return title_string

def extract_video_ids_and_contents(master_DF):
    '''extracts video ids from urls to read in later
    adds column to master DF for faster processing later'''

    master_DF['Video IDs'] = master_DF['link'].apply(extract_ids)
    master_DF['video_id'] = master_DF['link'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else "DROP")
    master_DF.drop_duplicates(subset=['video_id'], inplace=True)
    master_DF.drop(master_DF[master_DF['video_id'] == "DROP"].index, inplace = True)
    master_DF.head()
    master_DF.to_csv("VideosAndIDs.csv")
    master_DF.to_csv("VideosAndIDsAndText.csv")
    master_DF['title'] = master_DF['video_id'].apply(lambda x: pd.Series(extract_title(x)))
    print("did titles, this part took", (end_calcs-start_calcs)/60.0, "minutes to run")
    master_DF['text'] = master_DF['video_id'].apply(lambda x: pd.Series(extract_transcript(x)))
    master_DF['modality'] = "speech"
    master_DF.to_csv("FinalDF.csv")
    return master_DF

def extract_ids(url):
    pattern = r'(?:\?v=|&v=|/)([-\w]+)(?:&|$)'
    try:
        match = re.search(pattern, url)
        try:
            if match:
                #print(match)
                video_id = match.group(1)
            else:
                # Handle case where pattern is not found
                video_id = None
        except:
            print(video_id)
            print(url)
    except:
        video_id = "DROP"
    return video_id

def extract_title(video_id):
    link = "https://www.youtube.com/watch?v="+video_id
    try:
        title = YouTube(link).title
        split_string = re.split(r'\s*\|\s*', title)[0]
    except:
        split_string = "TITLE NOT AVAILABLE"
    return split_string

# Function to extract the transcript from a video ID
def extract_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine the text from each caption into a single string
        raw_transcript_text = " ".join(segment["text"] for segment in transcript)
        raw_transcript_text = "".join(remove_unprintable_chars(raw_transcript_text))
        if len(raw_transcript_text) > 32000:
            raw_transcript_text = raw_transcript_text[0:32000]
        #print(type(raw_transcript_text))
        # Return the video title and transcript
        return raw_transcript_text
    except:
        # Handle exceptions (e.g. invalid video ID or missing transcript)
        return "Error: Unable to extract transcript."

def extract_clean_text(raw_text):
    text = clean_text(raw_text)
    return text

def clean_text(text_string):
    text = re.sub("[\[].*?[\]]", " ", text_string)
    text = re.sub("[\(\{].*?[\}\)]"," ",text)
    text = re.sub("http[:/A-Za-z0-9\.\-]+", " web_link ", text) #removing urls
    text = re.sub("www\.[/A-Za-z0-9\.\-]+", " web_link ", text) #removing urls
    text = re.sub("@[A-Za-z0-9_\.\-]+", " at_tag ", text) #removing usernames (maybe want to add back in later if you look at behavior when atting versus hashtags/general posting)
    text = re.sub("#[A-Za-z0-9_\.]+", " hash_tag ", text) #removing hashtags with text after
    text = re.sub("#", "hashtag ", text) #removing standalone hashtags
    text = text.lower().split() #can use if you want idk
    for i in range(len(text)):
        while len(text[i]) > 0 and text[i][0] in END:
            #print(text[i])
            text[i] = text[i][1:]
            #print(text[i])
            #if len(text[i]) == 0:
            #    print("zero")
            #elif len(text[i]) > 0:
            #    print(text[i])
            #print("final word: " + str(text[i]))
        while len(text[i]) > 0 and text[i][-1] in END:
            #print(text[i])
            text[i] = text[i][:-1]
            #print(text[i])
            #print("final word: " + str(text[i]))
            #if len(text[i]) == 0:
            #    print("zero")

    return text

def remove_unprintable_chars(text_as_string):
    printables = list(string.printable)
    filtered_printable = "".join(filter(lambda x: x in printables, text_as_string))
    return filtered_printable

def clean_transcript(text):
    text = text.strip().lower() #lowercase tweets
    text = text.replace("\&amp;"," and ")
    text = text.replace("\&[a-zA-Z0-9#];"," and ")
    text = re.sub("&amp", " and ", text)
    text = re.sub("&", " and ", text)
    text = re.sub("\+", " plus ", text)
    text = re.sub("[’’“”’’‘ʺ]", "'", text)
    text = re.sub("[\$](?=[\d]+)", " dollars ", text)
    text = re.sub("^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$","phone_number",text)
    text = re.sub("\.{2,}", " ", text)
    text = re.sub("\-{2,}", " ", text)
    text = re.sub("%"," percent ",text)
    text = re.sub(r'[\u4e00-\u9fff]+', " ", text)
    text = re.sub("[A-Za-z0-9]{32,}","string_over_32_chars", text)
    text = re.sub("\d{9,}","nine_or_more_digits_string", text)
    text = re.sub("\d{5,9}","five_to_nine_digits_string", text)
    text = re.sub("(?<!\w)-(?=[\d]+)", "negative_amount ", text) #could also be negative #PUT NEGATIVE LOOKBACK FOR A NUMBER BEFORE

    text = re.sub("'(?=\s+)", " ", text) #for random just ' characters
    text = re.sub("a{3,}(?<=a)","a_elongated_",text)
    text = re.sub("b{3,}(?<=b)","b_elongated_",text)
    text = re.sub("c{3,}(?<=c)","c_elongated_",text)
    text = re.sub("d{3,}(?<=d)","d_elongated_",text)
    text = re.sub("e{3,}(?<=e)","e_elongated_",text)
    text = re.sub("f{3,}(?<=f)","f_elongated_",text)
    text = re.sub("g{3,}(?<=g)","g_elongated_",text)
    text = re.sub("h{3,}(?<=h)","h_elongated_",text)
    text = re.sub("i{3,}(?<=i)","i_elongated_",text)
    text = re.sub("j{3,}(?<=j)","j_elongated_",text)
    text = re.sub("k{3,}(?<=k)","k_elongated_",text)
    text = re.sub("l{3,}(?<=l)","l_elongated_",text)
    text = re.sub("m{3,}(?<=m)","m_elongated_",text)
    text = re.sub("n{3,}(?<=n)","n_elongated_",text)
    text = re.sub("o{3,}(?<=o)","o_elongated_",text)
    text = re.sub("p{3,}(?<=p)","p_elongated_",text)
    text = re.sub("q{3,}(?<=q)","q_elongated_",text)
    text = re.sub("r{3,}(?<=r)","r_elongated_",text)
    text = re.sub("s{3,}(?<=s)","s_elongated_",text)
    text = re.sub("t{3,}(?<=t)","t_elongated_",text)
    text = re.sub("u{3,}(?<=u)","u_elongated_",text)
    text = re.sub("v{3,}(?<=v)","v_elongated_",text)
    text = re.sub("w{3,}(?<=w)","w_elongated_",text)
    text = re.sub("x{3,}(?<=x)","x_elongated_",text)
    text = re.sub("y{3,}(?<=y)","y_elongated_",text)
    text = re.sub("z{3,}(?<=z)","z_elongated_",text)

    #design choice: separate am and pm and a.m. and p.m. from the digits before them, same with units of weight (was difficult to set tokenizer to do this automatically, may have missed some)
    text = re.sub("(?<=[\d])p\.*m\.*", " pm_time ", text)
    text = re.sub("(?<=[\d])a\.*m\.*", " am_time ", text)
    text = re.sub("(?<=\d)lbs(?!\w)", " lbs", text)
    text = re.sub("(?<=\d)lb(?!\w)", " lb", text)
    text = re.sub("(?<=\d)kgs(?!\w)", " kgs", text)
    text = re.sub("(?<=\d)kg(?!\w)", " kg", text)
    text = re.sub("(?<=\d)mins(?!\w)", " mins", text)
    text = re.sub("(?<=\d)min(?!\w)", " min", text)
    text = re.sub("(?<=\d)g(?!\w)", " g", text)
    text = re.sub("(?<=\d)oz(?!\w)", " oz", text)
    text = re.sub("(?<=\d)ft(?!\w)", " ft", text)
    text = re.sub("(?<=\d)in(?!\w)", " in", text)
    text = re.sub("(?<=\d)k(?!\w)", " k", text)

    text = re.sub("\S+\.com"," web_link " , text)
    text = re.sub(",", " ", text)

    text = re.sub(r"([a-zA-Z0-9]{2,})\.([a-zA-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]{2,})\.([a-zA-Z0-9]{2,})", r"\1 \2", text)

    return text

def change_or_make_path(path_addition):
    if os.path.exists(os.getcwd()+"/"+path_addition):
        os.chdir(os.getcwd()+"/"+path_addition)
    else:
        os.mkdir(path_addition)
        os.chdir(os.getcwd()+"/"+path_addition)

# need to write it all to a file: author, kind,
def make_fina_data_file():
    INPUT_FOR_CSV = []
    DONE = False
    while DONE == False:
        file_name = str(input("please enter a file name to be combined.\n]n be sure to not add quotes but do include the extension. type DONE when done: "))
        if file_name == "DONE" or file_name == "Done" or file_name == "done":
            DONE = True
        else:
            INPUT_FOR_CSV.append(file_name)
    FINAL_CSVS_TO_MERGE=read_csvs_make_DFs_List(INPUT_FOR_CSV)
    FINAL_DF_TO_WRITE, LINKS_ALL = concat_all_dfs(DF_LIST)
    FINAL_DF_TO_WRITE.to_csv("All Video Info And Text.csv")
    return INPUT_FOR_DFS

def main():
    start_calcs = time.time()
    #INPUT_FOR_DFS = gather_file_names()
    master_list_of_dfs = read_csvs_make_DFs_List(FILE_NAME_LIST)
    big_df = concat_all_dfs(master_list_of_dfs)
    change_or_make_path(TIMESTRING)
    big_df = extract_video_ids_and_contents(big_df)
    #extract_transcript_text(big_df)
    if LAST == "Yes":
        make_final_data_file()
    end_calcs = time.time()
    print("program code took: ", (end_calcs-start_calcs)/60.0, "minutes to run")

main()
