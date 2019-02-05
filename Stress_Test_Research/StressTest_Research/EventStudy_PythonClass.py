
# TODO: Create Python Class Library
#DONE: Phase 1: Event Data Acquisition.
    #A. Web Scrape Events Data (Done in R already) +++++++++++++++Complete           
        ## leverage Rpy to run script and import data frame?+++++++++++++++Complete
        ## Events , FRB, CEBS, EBA, IMF, FSB+++++++++++++++Complete
        #FSB events not being scrapped properly. 2017 and 2018 events.
        
        ## Normalize and Combine Events with proper tagging.+++++++++++++++Complete
        ## Provide Event Type out of Categories, Announcement, Results, Schedule, Etc. +++++++++++++++Complete
        

#TODO: Phase 2: Indicies and Identifier Acquisition

    #A. World Returns, Daily, Monthly, Consituents. - Inprogress
        ## Fic Codes, Country Level,  - Inprogress

    #B. Region Level Index Returns - Inprogress
        #Ticker, GVKEY, PERMCO, PERMNO - Inprogress
    #C. Sector Level Index Returns - Inprogress
     #Ticker, GVKEY, PERMCO, PERMNO - Inprogress
    #D. Bank Level Returns
        #Get info for bank participants in Euro, Asia, US, etc.
        #Ticker, GVKEY, PERMCO, PERMNO
        #EBA ~91 banks in 2010
        #US ~41 banks
        
    
#TODO: Phase 3: Market Data Acquisition
    #A. Equities
    #B. CDS Spreads
    #C. Bond Returns
    #D. IBES Recommendations
    #E. Options Spreads

#TODO: Phase 4: Banking Characteristics Acquisition
    #Capital Ratios
    #PPNR componets
    #Shadow Banking Elements
    #FR-14 elements?

#TODO: Phase 5: Event Study Analysis

#TODO: Phase 6: Cluster Analysis

#TODO: Phase 7: Scenario Based Balance Sheet Projections

#TODO: Phase 8: Results Analysis.


#import rpy2.rinterface
#import wrds


import os
os.environ['R_HOME'] = '/anaconda3/lib/R'

import gc
gc.enable()
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
# Hide warnings if there are any
import warnings
warnings.filterwarnings('ignore')

pandas2ri.activate()

#%load_ext rpy2.ipython

#TODO: Put in logging and Timers into each method.

#Variables for FSB is Year Range and Policy Areas

def fsb_events(yearrange = "2000:2019", policyareas = '''"non-bank-financial-intermediation", "vulnerabilities-assessments"'''):
    print("Starting FSB Events Web Scrape")
    
    fsb_events = ro.r('''
    if (!require("rvest")) {
      install.packages("rvest")
        }
    library("rvest")
    
    if (!require("devtools")) {
      install.packages("devtools")   
    }

   
    if (!require("feedeR")) {
      devtools::install_github("animalnexus/feedr") 
        }
    library("feedeR")
    
    if (!require("stringr")) {
      install.packages("stringr")   
    }
    library(stringr)


    #FSB Global Shadow Banking Report.
    #Policy Area : shadow-banking was scrubbed from website.
    #Working Policy Areas: non-bank-financial-intermediation, vulnerabilities-assessments

    yearrange = '''+ yearrange +'''
    policyareas = c('''+ policyareas +''')

    rss_df_1 = data.frame()
    ytmp = c()

    for (y in policyareas){

          #print(paste0("Policy Area: ",y))
          #Policy Areas List
          ytmp = c(ytmp,y)

          #Tmp rss iteration for policy area
          rss_df_tmp = data.frame()

      for (i in (yearrange))
          {
          
            for (j in 1:12){  
            url =paste0("http://www.fsb.org/",i,"/",j,"/feed/?policy_area=",y)
            urlrss = feed.extract(url)
            #print(url)
            #TODO:May need verbose switch
            #print(urlrss$items)

            if(nrow(urlrss$items) > 0) {
              names(urlrss$items)[4] = "category"
              urlrss$items[,4] = y
              #urlrss$items$category = y
              urlrss$items$country = "Global"
              rss_df_tmp = rbind(rss_df_tmp,urlrss$items)
              #print(dim(rss_df_tmp))
            }
            
            }

    
          }
     #Dup Checking and tagging for removal
     rss_df_1 = rbind(rss_df_1,rss_df_tmp)
     #dim(rss_df_1)
     dupidx = duplicated(rss_df_1[,1:3])
     rss_df_1[dupidx,4] = str_c(ytmp, collapse = ", ")
    }

#Dup Removal
    #detach("package:feedeR", unload=TRUE)
    library(dplyr)
    rss_df_1 <- 
    rss_df_1 %>% group_by(title,date) %>% filter(duplicated(title,date) | n()==1)
    
    rss_df_1
    ''')
    #May need to consider timezones
    fsb_events["date"] = pd.to_datetime( fsb_events["date"])
    fsb_events["source"] = "FSB"
    #.dt.strftime('%m/%d/%Y')
    print("Ending FSB Events Web Scrape")

    return(fsb_events)



#Method to Parse EBA Events.
def eba_st_events(yearrange = "2009:2019", urlbase = '''"https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-stress-testing"'''):
#EBA Stress Testing
    print("Starting EBA Stress Test Events Web Scrape")

    eba_events = ro.r('''
    if (!require("rvest")) {
      install.packages("rvest")
        }
    library("rvest")

    EBA_StressTest = data.frame()

    urlbase = '''+ urlbase + '''

    ebayears = '''+ yearrange + '''

    for (i in ebayears){
    
      #Need to implement try catch logic
      
      tmpurl = paste0(urlbase,"/",i)
      #May need verbose switch
      #print(tmpurl)
      if(!is.na(tryCatch( tmpurl %>% as.character() %>% read_html(), error = function(e){NA})))
     {
      
      webpage = read_html(tmpurl)
      #scraptmp = html_nodes(webpage, "dl")
  
      #Short Title
      title = webpage %>% html_nodes('dl') %>% html_node("a") %>% html_text()
      #Description
      desc = webpage %>% html_nodes('dl') %>% html_node("dd") %>% html_text()
      #Date
      #date = as.Date(webpage %>% html_nodes('dl') %>% html_node("dd.TLDate") %>% html_text(), format = "%d/%m/%Y")
      date = webpage %>% html_nodes('dl') %>% html_node("dd.TLDate") %>% html_text()
      #URL
      url = paste0("https://www.eba.europa.eu", webpage %>% html_nodes('dl') %>% html_node("a") %>% html_attr("href"))

      tmpobj = data.frame(title,desc,date,url, country = "Europe", category = "StressTest") 
      if(nrow(tmpobj) > 0) {
              
               EBA_StressTest = rbind.data.frame(EBA_StressTest, tmpobj)
            }
     }
}

    EBA_StressTest
    ''')
    #May need to consider timezones
    eba_events["date"] = pd.to_datetime(eba_events["date"])
    eba_events["source"] = "EBA"
    eba_events["source"][eba_events["title"].str.contains("CEBS")] = "CEBS"
    eba_events["source"][eba_events["title"].str.contains("Results of 2010 EU wide stress testing exercise")] = "CEBS"
    print("Ending EBA Stress Test Events Web Scrape")

    #.dt.strftime('%m/%d/%Y')
    return(eba_events)



#Press Releases and News
def eba_pr_events(urlbase = '''"https://www.eba.europa.eu/news-press/news"'''):
    print("Starting EBA Press Release Events Web Scrape")

    eba_events = ro.r('''
    if (!require("rvest")) {
      install.packages("rvest")
        }
    library("rvest")
    
    
    urlpresssite =  '''+ urlbase + '''
    EBA_PressNews = data.frame()
    webpage = read_html(urlpresssite)


    #Short Title
    title = webpage %>% html_nodes("[class = ListItems]") %>% html_nodes('li') %>% html_node("[class = Title]") %>% html_text()
    #Description
    desc = webpage %>% html_nodes("[class = ListItems]") %>% html_nodes('li') %>% html_node("[class = Info]") %>%  html_node("p") %>% html_text()
    #Date
    #date = as.Date(sapply(strsplit(webpage %>% html_nodes("[class = ListItems]") %>% html_nodes('li') %>% html_node("[class = Info]") %>%  html_node("[class = Date]") %>% html_text(), "-"), head , 1) , format = "%d/%m/%Y")
    date = sapply(strsplit(webpage %>% html_nodes("[class = ListItems]") %>% html_nodes('li') %>% html_node("[class = Info]") %>%  html_node("[class = Date]") %>% html_text(), "-"), head , 1)
    #URL
    
    #May need to be dynamic string
    url = paste0("https://www.eba.europa.eu", webpage %>% html_nodes("[class = ListItems]") %>% html_nodes('li') %>% html_node("[class = Title]") %>% html_attr("href"))

    tmpobj = data.frame(title,desc,date,url, country = "Europe", category = "PressRelease") 

    EBA_PressNews = rbind.data.frame(EBA_PressNews, tmpobj)
    
    ''')
     #May need to consider timezones
    eba_events["date"] = pd.to_datetime(eba_events["date"])
    eba_events["source"] = "EBA"
    eba_events["source"][eba_events["title"].str.contains("CEBS")] = "CEBS"
    eba_events["source"][eba_events["title"].str.contains("Results of 2010 EU wide stress testing exercise")] = "CEBS"
    #.dt.strftime('%m/%d/%Y')
    print("Ending EBA Press Release Events Web Scrape")

    return(eba_events)
    
    


#IMF FSAP
def imf_fsap_events(urlbase = '''"https://www.imf.org/en/Publications/Search?series=IMF+Staff+Country+Reports&when=After&title=Financial+System+Stability+Assessment&series=Global%20Financial%20Stability%20Report%20&series=Technical%20Notes%20and%20Manuals"'''):
    imf_fsap_events = ro.r('''
    if (!require("rvest")) {
          install.packages("rvest")
            }
        library("rvest")
    if (!require("stringr")) {
          install.packages("stringr")   
        }
        library(stringr)
    #if (!require("XML")) {
    #      install.packages("XML")
    #        }
    #library(XML)
    
    url = '''+ urlbase +'''
    #Switch for Verbosity
    #url = "https://www.imf.org/en/Publications/Search?series=IMF+Staff+Country+Reports&when=After&title=Financial+System+Stability+Assessment&series=Global%20Financial%20Stability%20Report%20&series=Technical%20Notes%20and%20Manuals"
    #print(url)
    webpage = read_html(url)
    #Pages variable
    pages = webpage %>% html_nodes('p.pages') %>% html_text()
    pages_clean = gsub("[\r\n]", "", pages[1])
    pages_max = as.numeric(gsub("of ","",str_extract(pages_clean, "of [0-9]+")))

    IMF_FSAP_Events = data.frame()

    for (i in 1:pages_max){

      url_tmp = paste0(url,"&page=",i)
      webpage = read_html(url_tmp)
      Title = str_trim(gsub("Title:","",gsub("\\\s+", " ", str_trim(webpage %>% html_nodes("[class = search-results]") %>% html_nodes('[class = \"result-row pub-row\"]') %>% html_node("h6") %>% html_text()))))

      Country = str_trim(sapply(str_split(Title,":"),head,1))

      Author = str_trim(gsub("Author:","",gsub("\\\s+", " ", str_trim(webpage %>% html_nodes("[class = search-results]") %>% html_nodes('[class = \"result-row pub-row\"]') %>% html_node("p.author") %>% html_text()))))

      Series = str_trim(gsub("Series:","",gsub("\\\s+", " ", str_trim(webpage %>% html_nodes("[class = search-results]") %>% html_nodes('[class = \"result-row pub-row\"]') %>% html_node("p:nth-child(3)") %>% html_text()))))

      Date = as.character(as.Date(str_trim(gsub("Date:","",gsub("\\\s+", " ", str_trim(webpage %>% html_nodes("[class = search-results]") %>% html_nodes('[class = \"result-row pub-row\"]') %>% html_node("p:nth-child(4)") %>% html_text())))), format = "%B %d, %Y"))
      #Date = str_trim(gsub("Date:","",gsub("\\\s+", " ", str_trim(webpage %>% html_nodes("[class = search-results]") %>% html_nodes('[class = \"result-row pub-row\"]') %>% html_node("p:nth-child(4)") %>% html_text()))))
      tmp_df = data.frame(Title,Date,Country, category = "FSSA",Author, Series)
      IMF_FSAP_Events = rbind(IMF_FSAP_Events,tmp_df)
                            }
      IMF_FSAP_Events
      
        ''')
    #May need to consider timezones
    imf_fsap_events["Date"] = pd.to_datetime(imf_fsap_events["Date"])
    imf_fsap_events["source"] = "IMF"
    imf_fsap_events.columns = imf_fsap_events.columns.str.lower()
    #.dt.strftime('%m/%d/%Y')
    return(imf_fsap_events)



def frb_events(urlbase = 'https://www.federalreserve.gov', st_urlbase = 'https://www.federalreserve.gov/supervisionreg/',ccaryearrange = "2011:2018", dfastyearrange = "2013:2018" ):
 
    frb_events = ro.r('''
        if (!require("rvest")) {
          install.packages("rvest")
            }
        library("rvest")
        if (!require("stringr")) {
          install.packages("stringr")   
        }
        library(stringr)
        
        urlbase = 'https://www.federalreserve.gov'
        urlbase2 = 'https://www.federalreserve.gov/supervisionreg/'
        url = 'https://www.federalreserve.gov/supervisionreg/ccar.htm'

        #CCAR
        ccar_yearrange = 2011:2018
        ccarpages = c(paste(urlbase2,"ccar-",ccar_yearrange,".htm",sep = ""))
        #2017 rule
        ccarpages = gsub("ccar-2017.htm","ccar-2017-archive.htm", ccarpages)
        #2018 rule
        ccarpages = gsub("ccar-2018.htm","ccar.htm", ccarpages)
        ccarpages = data.frame(urls = ccarpages, category = "CCAR")
  
        #DFAST  
        dfastyearrange = 2013:2018
        dfastpages = c(paste(urlbase2,"dfast-",dfastyearrange,".htm",sep = ""))
        #2018 rule
        dfastpages = gsub("dfast-2018.htm","dfa-stress-tests.htm", dfastpages)
        dfastpages = data.frame(urls = dfastpages, category = "DFAST")

        #Combine into dataframe
        frbpages = rbind(ccarpages,dfastpages)
        frbpages$urls = as.character(frbpages$urls)
        frbpages$category = as.character(frbpages$category)

        #Get title and urls to scrape
        frb_df = data.frame()
        for( i in 1:nrow(frbpages)) 
                {
                  #Verbose Switch
                  #print(frbpages$urls[i])
                  webpage = read_html(frbpages$urls[i])
                  title = str_trim(webpage %>% html_nodes('[id = article]') %>% html_nodes(xpath = '//*[@id="article"]/ul[1]') %>% html_nodes('a') %>% html_text())
                  #url = paste0(urlbase, webpage %>% html_nodes('[id = article]') %>% html_nodes(xpath = '//*[@id="article"]/ul[1]') %>% html_nodes('a') %>% html_attr("href"))
                  url = webpage %>% html_nodes('[id = article]') %>% html_nodes(xpath = '//*[@id="article"]/ul[1]') %>% html_nodes('a') %>% html_attr("href")
                  frb_df_tmp = data.frame(title,url,category = frbpages$category[i])
                  frb_df = rbind(frb_df,frb_df_tmp)
                }

        frb_df$title = as.character(frb_df$title)
        frb_df$url = as.character(frb_df$url)
        frb_df$category = as.character(frb_df$category)
        frb_df$url[grep("^/newsevents/pressreleases", frb_df$url)] = paste(urlbase,frb_df$url[grep("^/newsevents/pressreleases", frb_df$url)],sep = "")

        #Get Date, Time and Title for Press Releases
        frb_df2 = data.frame()

        for(i in 1:nrow(frb_df))
        {
          #Verbose Switch
          #print(frb_df$title[i])
          #print(frb_df$url[i])
          #print(frb_df$category[i])
  
          webpage1 = read_html(frb_df$url[i])

      releasedate = webpage1 %>% html_nodes('[id = article]') %>% html_nodes('p.article__time') %>% html_text()
      title = webpage1 %>% html_nodes('[id = article]') %>% html_nodes('h3.title') %>% html_text()
      releaseTime = str_trim(gsub("For release at", "", webpage1 %>% html_nodes('[id = article]') %>% html_nodes('p.releaseTime') %>% html_text()))
      lastUpdate = str_trim(gsub("Last Update:","",gsub("\\\s+", " ", str_trim(webpage1 %>% html_nodes('[id = lastUpdate]') %>%  html_text()))))
  
      frb_df2_tmp = data.frame(title,releasedate,releaseTime,lastUpdate, url = frb_df$url[i], country = "USA", category = frb_df$category[i])
      frb_df2 = rbind(frb_df2, frb_df2_tmp)
            }

        frb_df2
    ''')
    frb_events['date1'] = frb_events["releasedate"].astype(str) + " " + frb_events["releaseTime"].astype(str).replace("For immediate release","9:00 a.m. EST")
    frb_events['date1'] = frb_events['date1'].str.replace("a.m.","AM").str.replace("p.m.","PM") 
    frb_events['date'] = pd.to_datetime(frb_events['date1'], utc = True).dt.tz_convert("UTC").dt.tz_convert("US/Eastern")
    frb_events['source'] = "FRB"
    return(frb_events)


#Wrapper Function to get events
def getevents_data(eventscolumnlist = ['title','date','country','category','source']
                   , source = ["fsb","eba","imf","frb"]):
    #Make switched and arguements to select events to gather and return
    events_combined = pd.DataFrame({'empty' : []})
    tmp_raw = []
    if "fsb" in source:
        print("Started: Scraping Events from FSB")
        tmp_raw = fsb_events()
        print("Finished: Scraping Events from FSB")
        print("Adding Events to ResultsDataframe")
        if not events_combined.empty:
            events_combined = pd.concat([events_combined,tmp_raw[eventscolumnlist]])
        else:
            events_combined = tmp_raw[eventscolumnlist]
            
    if "eba" in source:                                
        print("Started: Scraping Events from EBA")
        print("Started: Scraping Stress Test Events from EBA")
        tmp_raw = eba_st_events()
        print("Finished: Scraping Stress Test Events from EBA")
        print("Adding Events to ResultsDataframe")
        if not events_combined.empty:
            events_combined = pd.concat([events_combined,tmp_raw[eventscolumnlist]])
        else:
            events_combined = tmp_raw[eventscolumnlist]
        
        print("Started: Scraping Press Release Events from EBA")
        tmp_raw = eba_pr_events()
        print("Finished: Scraping Press Release Events from EBA")
        print("Adding Events to ResultsDataframe")
        
        if not events_combined.empty:
            events_combined = pd.concat([events_combined,tmp_raw[eventscolumnlist]])
        else:
            events_combined = tmp_raw[eventscolumnlist]
                                
    if "imf" in source:
                                
        print("Started: Scraping FSAP Events from IMF")
        tmp_raw = imf_fsap_events()
        print("Finished: Scraping FSAP Events from IMF")
        print("Adding Events to ResultsDataframe")
        if not events_combined.empty:
            events_combined = pd.concat([events_combined,tmp_raw[eventscolumnlist]])
        else:
            events_combined = tmp_raw[eventscolumnlist]
    if "frb" in source:    
        print("Started: Scraping Stress Test Press Releases from FRB")
        tmp_raw = frb_events()
        print("Finished: Scraping Stress Test Press Releases from FRB")
        print("Adding Events to ResultsDataframe")
        if not events_combined.empty:
            events_combined = pd.concat([events_combined,tmp_raw[eventscolumnlist]])
        else:
            events_combined = tmp_raw[eventscolumnlist]
    

    print("Events Scraping Completed")
    events_combined = events_normalize(events_combined)
    events_combined = events_combined.reset_index()

    return(events_combined)


#Improve by using python dict
def events_normalize_annctype(events = None, source = ["eba","fsb","imf","frb"]):

    print("Check for Column annctype")
    if 'annctype' not in events.columns:
        print("Creating Column annctype")
        events['annctype'] = ""



    if "eba" in source:
        #print("EBA")
        print("EBA tagging rules")
        # Focus on the Stress Testing Events.
        # Remove Press Releases not related.

        sourcetuple = ('EBA','CEBS')
        # Announcements
        AnnouncementStrings = "Announce|announce|launch|update|next|publishes|Publishes|" \
                              " Publishes|publish|consultation|monitoring exercise" \
                              "|Consult|consult|acknowledges|Survey|Speech|speech|survey" \
                              "|Report on|report on|paper on| Comments| comment"

        StressTest_terms = "stress test|stress|CET1|instruments|transparency exercise" \
                           "|Tier 1|Stress Test|EU banks| EU bank|Euro-wide|EU-wide" \
                           "|capital|capital requirements|supervisory"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(AnnouncementStrings))
                           & (events['title'].str.contains(StressTest_terms))] = "Announcement"

        # Statement
        StatementStrings = "statement|press release|state of play"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(StatementStrings))
                           & (events['title'].str.contains(StressTest_terms))] = "Statement"

        # Methodology
        MethodologyStrings = "methodology|templates|infographic"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(MethodologyStrings))
                           & (events['title'].str.contains(StressTest_terms))] = "Methodology"

        # Results
        ResultsStrings = "result|Results"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(ResultsStrings))
                           & (events['title'].str.contains(StressTest_terms))] = "Results"

        # Recommendation
        RecommendationStrings = "Recommendation|recommendation"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(RecommendationStrings))
                           & (events['title'].str.contains(StressTest_terms))] = "Recommendation"

        print("Removing unnecessary events from dataframe")
        events = events.drop(events[(events['source'].str.endswith(sourcetuple)) & (events['annctype'] == "")].index)
        print("EBA Events normalize complete")




    if "fsb" in source:
        print("FSB tagging rules")

        sourcetuple = ('FSB')
        # Results
        ResultsStrings = "Global Shadow Banking Monitoring Report"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                      & (events['annctype'] == "")
                      & (events['title'].str.contains(ResultsStrings))] = "Results"

        #Recommendations
        RecommendationStrings = "Recommendation|recommends|recommendation"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(RecommendationStrings))] = "Recommendation"

        # Announcements
        AnnouncementStrings = "Plenary|plenary|Implementation|note|shadow banking|RCG|FSB|G20"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(AnnouncementStrings))] = "Announcement"

        # Statement
        StatementStrings = "Chair|Progress|progress|discusses"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(StatementStrings))] = "Statement"

        # Methodology
        MethodologyStrings = "Instruction|framework|Standards|guideline|Guideline"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(MethodologyStrings))] = "Methodology"

        print("Tagging remaining as announcement on dataframe")
        events['annctype'][(events['source'].str.endswith(sourcetuple)) & (events['annctype'] == "")] = "Announcement"
        print("FSB Events normalize complete")

    if "imf" in source:
        print("IMF tagging rules")
        sourcetuple  = ('IMF')
        # Methodology
        MethodologyStrings = "Technical Note| Publication"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(MethodologyStrings))] = "Methodology"

        print("Tagging remaining as results on dataframe")
        events['annctype'][(events['source'].str.endswith(sourcetuple)) & (events['annctype'] == "")] = "Results"

        print("IMF Events normalize complete")

    if "frb" in source:
        print("FRB tagging rules")
        sourcetuple = ('FRB')
        # Results
        ResultsStrings = "completes Comprehensive Capital Analysis and Review|announces summary results|releases results|releases summary results|announces results"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['annctype'] == "")
                           & (events['title'].str.contains(ResultsStrings))] = "Results"

        # Methodology
        MethodologyStrings = "releases scenarios|methodology|released instructions|Methodology|releases supervisory|scenarios|announces finalized"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(MethodologyStrings))] = "Methodology"

        # Announcement
        AnnouncementStrings = "schedule for results|announces schedule for|announces that|announces it|launches"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(AnnouncementStrings))] = "Announcement"
        # Statement
        StatementStrings = "Publishes|publishes|statement|statements|Statement|issues|issue|seek comment|approves|releases paper|invites comment|releases guidance|proposes rule"
        events['annctype'][(events['source'].str.endswith(sourcetuple))
                           & (events['title'].str.contains(StatementStrings))] = "Statement"

        print("Tagging remaining as results on dataframe")
        events['annctype'][(events['source'].str.endswith(sourcetuple)) & (events['annctype'] == "")] = "Announcement"
        print("FRB Events normalize complete")
        return(events)


def DeAgg_Events_Union(events_cleaned, BankingUnionColumn="BankingUnion", BankUnion_dict={'BankingUnion': ["European Union", "EuroZone", "CEMAC", "ECCU","European Union","Global"],
                                       'findstr_column': ["source", "country", "country", "country","country","country"],
                                       'findstr': ["EBA", "Euro Area Policies", "Central African Economic",
                                                   "Eastern Caribbean Currency Union","Europe","Global"],
                                       'matchtype': ["==", "contains", "contains", "contains","==","=="],
                                       'countries': [["Austria", "Italy", "Belgium", "Latvia", "Bulgaria", "Lithuania", "Croatia",
                                            "Luxembourg", "Cyprus", "Malta", "Czechia", "Netherlands", "Denmark",
                                            "Poland","Estonia", "Portugal", "Finland", "Romania", "France", "Slovakia",
                                            "Germany","Slovenia", "Greece", "Spain", "Hungary", "Sweden", "Ireland","United Kingdom"],
                                           ["Austria", "Belgium", "Cyprus", "Estonia", "Finland", "France", "Germany",
                                            "Greece", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Portugal", "Slovakia", "Slovenia", "Spain"],
                                           ["Cameroon", "Central African Rep.", "Chad", "Equatorial Guinea", "Gabon","Congo"],
                                           ["Antigua and Barbuda", "Dominica", "Grenada", "Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines"],
                                            ["Austria", "Italy", "Belgium", "Latvia", "Bulgaria", "Lithuania",
                                                      "Croatia",
                                                      "Luxembourg", "Cyprus", "Malta", "Czechia", "Netherlands",
                                                      "Denmark",
                                                      "Poland", "Estonia", "Portugal", "Finland", "Romania", "France",
                                                      "Slovakia",
                                                      "Germany", "Slovenia", "Greece", "Spain", "Hungary", "Sweden",
                                                      "Ireland", "United Kingdom"],
                                                     []
                                                     ]}, CombineFrame = False):
    print("Generating Known Banking Unions")
    # Create a Data Frame, Join to UN and focus on unmatched.

    if BankingUnionColumn not in events_cleaned.columns:
        print("Creating new column: Banking Union")
        events_cleaned[BankingUnionColumn] = np.nan

    tmp_df = pd.DataFrame()
    for i in range(BankUnion_dict["BankingUnion"].__len__()):
         #print(i)
         tmp = pd.DataFrame()
         tmp2 = pd.DataFrame()
         if "contains" in BankUnion_dict["matchtype"][i]:
            print("Setting Banking Union: " + BankUnion_dict["BankingUnion"][i])
            events_cleaned[BankingUnionColumn][(events_cleaned[BankUnion_dict["findstr_column"][i]].str.contains(BankUnion_dict["findstr"][i]))] = BankUnion_dict["BankingUnion"][i]
            print("Get Subset of Rows to Deaggregate")
            tmp = events_cleaned[(events_cleaned[BankUnion_dict["findstr_column"][i]].str.contains(BankUnion_dict["findstr"][i]))]
         if "==" in BankUnion_dict["matchtype"][i]:
            print("Setting Banking Union: " + BankUnion_dict["BankingUnion"][i])
            events_cleaned[BankingUnionColumn][(events_cleaned[BankUnion_dict["findstr_column"][i]] == BankUnion_dict["findstr"][i])] = BankUnion_dict["BankingUnion"][i]
            print("Get Subset of Rows to Deaggregate")
            tmp = events_cleaned[(events_cleaned[BankUnion_dict["findstr_column"][i]] == BankUnion_dict["findstr"][i])]

         print("Repeat the rows the length of the country list")
         if BankUnion_dict["countries"][i].__len__() > 1:
            tmp2 = pd.concat([tmp] * BankUnion_dict["countries"][i].__len__())
            print("Applying Country names to Deaggregated Rows")
            tmp2["country"] = BankUnion_dict["countries"][i] * tmp.shape[0]
            print('Attached Deaggregated Rows to temp object')
            tmp_df = pd.concat([tmp_df,tmp2], ignore_index= True)

    if CombineFrame:
        tmp_df = pd.concat([events_cleaned,tmp_df], ignore_index=True)
        #tmp_df = tmp_df.reset_index()
    tmp_df = tmp_df.reset_index()
    return(tmp_df)



def get_CountryCodes(excelurl = ["http://unstats.un.org/unsd/tradekb/Attachment440.aspx?AttachmentType=1"]
                    ,jsonurl = ["https://comtrade.un.org/data/cache/reporterAreas.json","https://comtrade.un.org/data/cache/partnerAreas.json"]
                    ,weburl = ["https://www.nationsonline.org/oneworld/country_code_list.htm"]):

    print("Initializing Dataframe for CountryCodes")
    CountryCodes = pd.DataFrame()

    if len(excelurl) > 0:
        for url in excelurl:
            print("Getting Excel File from: " + url)
            CountryCodes_tmp = pd.read_excel(url)
            CountryCodes_tmp = CountryCodes_tmp[["Country Code", "Country Name, Abbreviation", "ISO2-digit Alpha", "ISO3-digit Alpha"]]
            CountryCodes_tmp["code_source"] = "UN_Excel"
            CountryCodes_tmp["orderpref"] = 1
            CountryCodes_tmp.columns = ["UN_CountryCode", "country", "ISO2", "ISO3","code_source","orderpref"]
            print("Adding Rows to Initialized Dataframe.")
            CountryCodes = pd.concat([CountryCodes, CountryCodes_tmp], ignore_index= True)

    if len(weburl) > 0:
        FICUNCodes[["UNCode_ID","country","FIC_2","FIC_3"]]
        for url in weburl:
            print("Getting Web Scrape from: " + url)
            import urllib3
            from bs4 import BeautifulSoup
            http = urllib3.PoolManager()
            response = http.request("GET", countrycodeurl)
            soup = BeautifulSoup(response.data)
            CountryCodes_tmp = pd.DataFrame()
            # FICUNCodes.columns = ["Country","FIC_2","FIC_3","UNCode_ID"]
            for tr in soup.find_all('tr')[2:]:
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    tmp = {"country": tds[1].text.strip(), "FIC_2": tds[2].text.strip(), "FIC_3": tds[3].text.strip(),
                           "UNCode_ID": tds[4].text.lstrip("0")}
                    CountryCodes_tmp = CountryCodes_tmp.append(tmp, ignore_index=True)
            CountryCodes_tmp = CountryCodes_tmp[["UNCode_ID","country","FIC_2","FIC_3"]]
            CountryCodes_tmp["code_source"] = "NationsOnline"
            CountryCodes_tmp["orderpref"] = 2
            CountryCodes_tmp.columns = ["UN_CountryCode", "country", "ISO2", "ISO3", "code_source", "orderpref"]
            print("Adding Rows to Initialized Dataframe.")
            CountryCodes = pd.concat([CountryCodes, CountryCodes_tmp])

    if len(jsonurl) > 0:
        for url in jsonurl:
            print("Getting Json from: "+ url )
            CountryCodes_tmp = pd.read_json(url,orient='columns')
            print("Transforming JSON to Pandas Dataframe.")
            CountryCodes_tmp = pd.read_json((CountryCodes_tmp['results']).to_json(), orient='index')
            CountryCodes_tmp["ISO2"] = np.nan
            CountryCodes_tmp["ISO3"] = np.nan
            CountryCodes_tmp["code_source"] = "UN_JSON"
            CountryCodes_tmp["orderpref"] = 3
            CountryCodes_tmp.columns = ["UN_CountryCode", "country", "ISO2", "ISO3", "code_source", "orderpref"]
            print("Adding Rows to Initialized Dataframe.")
            #UN Dataframe combined
            CountryCodes = pd.concat([CountryCodes,CountryCodes_tmp],ignore_index= True)




        print("Manually Adding Countries that are Missing")
        #Added Kosovo, Gurensey,Jersey

        print("Adding Kosovo")
        tmp = {"UN_CountryCode":"XKX","country":"Kosovo","ISO2":"XK","ISO3":"XKX","code_source":"Manual","orderpref": 4}
        CountryCodes = CountryCodes.append(tmp, ignore_index = True)



        print("Dropping Duplicates after Manual Add")
        #More logic required to take UN excel and json as preference before nations online

        CountryCodes = CountryCodes.sort_values(["country","orderpref"]).drop_duplicates(subset =["country"], keep="first")

        print("Resetting Index")
        CountryCodes = CountryCodes.reset_index()

        print("Returning UN_CountryCodes Dataframe.")
        return(CountryCodes)


def normalize_events_CountryCodes_UN(events_cleaned,UN_CountryCodes,findstr_dict = { 'source': ["IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF","IMF"] ,
                                                                                    'findstr':["Republic of ","Rep. of ","Hong Kong","Republic of China","United States","Tanzania","Czech Republic",
                                                                                   "Slovak","Latvia","Kazakhstan","Central African Republic","Kyrgyz","Netherlands","Turks and Caicos Islands",
                                                                                   "Bosnia and Herzegovina","Democratic Republic of the Congo","British Virgin Islands","Bahrain"
                                                                                    ,"Former Yugoslav Republic of Macedonia","Central African Economic and Monetary Community"],
                                                                                    'replacestr':["Rep. of ","","China, Hong Kong SAR","China","USA","United Rep. of Tanzania","Czechia","Slovakia",
                                                                                  "Latvia","Kazakhstan","Central African Rep.","Kyrgyzstan","Netherlands","Turks and Caicos Isds",
                                                                                  "Bosnia Herzegovina","Dem. Rep. of the Congo","Br. Virgin Isds","Bahrain","TFYR of Macedonia"
                                                                                        ,"Central African Economic and Monetary Community"],
                                                                                'rule':[1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                                                'ruleorder':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                                     columns_orig = ["title","date","country","category","source","annctype","BankingUnion"],
                                     iso3_ovr_dict = { 'country': ['Global','Europe','Euro Area Policies'],
                                                       'ISO3': ['Global','EUR','EURO']}):

    print("Creating Rule Set")
    normalize_dict_pd = pd.DataFrame.from_dict(findstr_dict)
    normalize_dict_pd = normalize_dict_pd.sort_values("ruleorder")
#    print("Sotring Original Columns")

    print("Initial Join to UN_CountryCodes")
    events_cleaned = events_cleaned.merge(UN_CountryCodes,left_on = 'country', right_on = 'country', how = 'left' )

    print("Applying Rules")
    for i in range(len(normalize_dict_pd)):
        if normalize_dict_pd.loc[i]["rule"] == 1:
            print("RuleOrder: " + str(normalize_dict_pd.loc[i]["ruleorder"])  + " Rule: " + str(normalize_dict_pd.loc[i]["rule"]) + " Source: " + normalize_dict_pd.loc[i]["source"])
            print(" Replace Country Title: " + normalize_dict_pd.loc[i]["findstr"] + " with: " + normalize_dict_pd.loc[i]["replacestr"])
            #Get Index Rows for matching Pattern
            idxrows = events_cleaned["country"][(pd.isnull(events_cleaned["ISO3"]))
                                                & (events_cleaned["source"] == normalize_dict_pd.loc[i]["source"])
                                                & (events_cleaned["country"].str.startswith(normalize_dict_pd.loc[i]["findstr"]))].index

            #Replace the String in subset
            events_cleaned["country"][idxrows] = events_cleaned["country"][idxrows].str.replace(normalize_dict_pd.loc[i]["findstr"], normalize_dict_pd.loc[i]["replacestr"])


            #Merge to the UNcodes to see how many resolve for next iteration
            events_cleaned = events_cleaned[columns_orig].merge(UN_CountryCodes, left_on='country', right_on='country',
                                                            how='left')

        if normalize_dict_pd.loc[i]["rule"] == 2:
            print("RuleOrder: " + str(normalize_dict_pd.loc[i]["ruleorder"]) + " Rule: " + str(normalize_dict_pd.loc[i]["rule"]) + " Source: " + normalize_dict_pd.loc[i]["source"])
            print(" Replace Country Title: " + normalize_dict_pd.loc[i]["findstr"] + " with: " + normalize_dict_pd.loc[i]["replacestr"])

            events_cleaned["country"][(pd.isnull(events_cleaned['ISO3'])) & (events_cleaned["source"] == normalize_dict_pd.loc[i]["source"]) & (
            events_cleaned["country"].str.contains(normalize_dict_pd.loc[i]["findstr"]))] = normalize_dict_pd.loc[i]["replacestr"]

    print("Final Merge with all Rules completed")
    events_cleaned = events_cleaned[columns_orig].merge(UN_CountryCodes, left_on='country', right_on='country',
                                                        how='left')
    print("Number of rows that did not match: ", events_cleaned["country"][pd.isnull(events_cleaned['ISO3'])].__len__())



    print("Dropping Index Column")
    events_cleaned = events_cleaned.drop("index", axis =1)
    print('Renaming Columns')
    #events_cleaned = events_cleaned.rename(columns = {"id" : "UN_CodeID","text":"UN_CountryName"})
    #print(events_cleaned.columns)
    #columns_orig.extend(["UNCode_id","UNCountry_Name"])
    #print(columns_orig)
    #events_cleaned.columns = columns_orig

    print("Filling ISO3 with Overrides")
    for i in range(iso3_ovr_dict["country"].__len__()):
        events_cleaned['ISO3'][events_cleaned['country'] ==  iso3_ovr_dict["country"][i]] = iso3_ovr_dict["ISO3"][i]


    return(events_cleaned)





def get_CountryIndices(events_cleaned, country_indicies_file = "wrdsWorldIndiciesFIC_Indiicies.csv",gvkey_column = "GVKEY",
                       wrds_username="fr497", password = "Fr056301", na_wrdstable = 'comp_na_daily_all.idx_index',glb_wrdstable = "comp_global.g_idx_index",
                        query_fields_tmp = ["conm", "gvkeyx", "idx13key", "idxcstflg", "idxstat", "indexcat", "indexgeo", "indexid","indextype", "indexval", "tic", "tici"],
                        indexgeo_ovr_dict = { 'qrycat': ['Global','EURO'],
                                                       'indexgeo': ['Global','EURO']},
                        RunWrdsAPI = True):

    print("Getting List of Countries from Events Data Frame")
    FIC_List = events_cleaned['ISO3'][pd.notnull(events_cleaned["ISO3"])].unique()
    print(FIC_List)

    print("Getting List of Generalized Regions and Banking Unions for manual mapping")
    MnlMap_List = events_cleaned["country"][(pd.notnull(events_cleaned["BankingUnion"])) & (pd.isnull(events_cleaned["ISO3"]))].drop_duplicates()

    print("Creating WRDS API query string for Global Indices based on Country Code")
    query_dict = {'Cat': ['FIC', 'FIC_NAM', 'EUR', 'EURO', "Global", "NAM_Indices"],
                  'select': [query_fields_tmp, query_fields_tmp, query_fields_tmp, query_fields_tmp, query_fields_tmp,
                             query_fields_tmp],
                  'from': [glb_wrdstable, na_wrdstable, glb_wrdstable, glb_wrdstable, glb_wrdstable, na_wrdstable],
                  'where': [["indexgeo in ('" + "','".join(FIC_List) + "')", "indextype in ('COMPOSITE','REGIONAL')",
                             "indexid not in ('ISLAMIC')"],
                            ["indexgeo in ('" + "','".join(FIC_List) + "')",
                             "indextype in ('COMPOSITE','REGIONAL','LGCAP')","conm similar to '%%(S&P|Nasdaq|NY Stock Exchange|Russell|NYSE)%%'"],
                            ["indextype in ('COMPOSITE', 'REGIONAL')", "indexgeo = 'EUR'",
                             "indexid not in ('ISLAMIC') "],
                            ["indextype in ('COMPOSITE', 'REGIONAL')", "indexval = 'EURO'",
                             "indexid not in ('ISLAMIC') "],
                            ["indexgeo is null", "indexid not in ('ISLAMIC')", "indexval similar to '%%(ALL)%%'",
                             "conm similar to '%%(Global|World|All)%%'"],
                            ["indextype in ('COMPOSITE','REGIONAL','LGCAP')",
                             "conm similar to '%%(S&P|Nasdaq|NY Stock Exchange|Russell|NYSE)%%'"]]
                  }

    if country_indicies_file is not None:
        print("Loading Country Indicies File: ",country_indicies_file )
        countryFICgvkey = pd.read_csv(country_indicies_file)
        print("Extracting GVKEYs for WRDS Pull")
        countryFICgvkey[gvkey_column] = countryFICgvkey[gvkey_column].astype(str)
        gvkeylist = countryFICgvkey[gvkey_column]
        print("Stringifying for WRDS API request")
        gvkeylist = "','".join(gvkeylist)
        query_dict["Cat"].append('StaticCountryFile')
        query_dict['select'].append(query_fields_tmp),
        query_dict['from'].append(glb_wrdstable)
        query_dict['where'].append(["gvkeyx in ('" + gvkeylist + "')"])



    wrds_query = []
    for i in range(query_dict["Cat"].__len__()):
        print(query_dict["Cat"][i])
        query_tmp = "select " + ",".join(query_dict["select"][i]) + ", '" + query_dict["from"][i] + "'" + " as source"+ ", '" + query_dict["Cat"][i] + "'" + " as qryCat"
        #print(query_tmp)
        query_tmp = query_tmp + " from "+ query_dict["from"][i]
        #print(query_tmp)
        query_tmp = query_tmp + " where " + " and ".join(query_dict["where"][i])
        #print(query_tmp)
        wrds_query.append(query_tmp)

    wrdsquery_final = " union ".join(wrds_query)


    if RunWrdsAPI:
        print("Initialize WRDS connection")
        import wrds
        db = wrds.Connection(wrds_username= wrds_username, password= password)
        #glb_wrds_indices = pd.DataFrame()
        print("Running WRDS API Query and retrieving data into Object")
        wrds_query_df = pd.DataFrame()
        wrds_query_df = db.raw_sql(wrdsquery_final)

        print("Deduplicating based on GVKEYs")
        wrds_query_df = wrds_query_df.ix[wrds_query_df[["gvkeyx","conm"]].drop_duplicates().index]
        print("Resetting Index")
        wrds_query_df = wrds_query_df.reset_index()
        print("Sorting Data")
        wrds_query_df = wrds_query_df.sort_values(["indexgeo","conm","idxstat","qrycat"], ascending = [1,1,1,1])

        print("Filling indexgeo with overrides")
        for i in range(indexgeo_ovr_dict["qrycat"].__len__()):
            wrds_query_df['indexgeo'][wrds_query_df['qrycat'] == indexgeo_ovr_dict["qrycat"][i]] = indexgeo_ovr_dict["indexgeo"][i]


    else:
        wrds_query_df = wrdsquery_final
        print(wrds_query_df)

    print("Returning Object")
    return(wrds_query_df)





#Get country regions
def get_country_regions(events_cleaned = None, regionurl = 'https://unstats.un.org/unsd/methodology/m49/', table_element = {'id': 'GeoGroupsENG'},
                        region_event_ovr_dict={'Global': 'World',
                                               'Europe': 'Western Europe',
                                               'Euro Area Policies': 'Western Europe',
                                               'Central African Economic and Monetary Community': 'Middle Africa',
                                               'Kosovo': 'Southern Europe',
                                               'Serbia and Montenegro': 'Southern Europe',
                                               'Eastern Caribbean Currency Union': 'Caribbean'},
                        webscrape = True, staticfile = 'world_country_regions_ref.csv'
                        ):

    if webscrape:
        import urllib3
        from bs4 import BeautifulSoup

        print("Initialize Connection to Web Source")

        http = urllib3.PoolManager()
        response = http.request("GET", regionurl)
        soup = BeautifulSoup(response.data)

        print("Scrape table")
        table = soup.find('table',table_element)

        print("Creating Data Frame")
        CountryRegion_tmp = pd.DataFrame()
        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')

            #print(tds)
        #    if len(tds) >= 3:
            if 'data-tt-id' in tr.attrs:
                datattid  = tr.attrs['data-tt-id']
            else:
                datatid = None
            if 'data-tt-parent-id' in tr.attrs:
                parent_id = tr.attrs['data-tt-parent-id']
            else:
                parent_id = None
            tmp = {"country/region": tds[0].text.strip(), "M49Code": tds[1].text.strip(), "ISO3": tds[2].text.strip(),"ddattid": datattid,"parent_id": parent_id}
            CountryRegion_tmp = CountryRegion_tmp.append(tmp, ignore_index=True)
    elif staticfile is not None:
        print("Static File Import :", staticfile)
        CountryRegion_tmp = pd.read_csv(staticfile)



    if events_cleaned is not None:
        print('Match country/regions to event countries')
        print('Match Ref Countries to self for parent regions')
        CountryRegion_tmp = CountryRegion_tmp.merge(CountryRegion_tmp[['M49Code','country/region']], left_on = 'parent_id', right_on = 'M49Code', how = 'left')
        events_countries_df = events_cleaned[["country", "ISO3"]].drop_duplicates()
        region_events_taged = events_countries_df.merge(CountryRegion_tmp, left_on='ISO3', right_on="ISO3", how='left')
        print("Merging object to self to get parent fields")
        print('Apply Overrides for Country/Region pairs')
        for k, v in region_event_ovr_dict.items():
            print(k, ' : ', v)
            region_events_taged['country/region_y'][region_events_taged['country'] == k] = v
            #region_events_taged[pd.isnull(region_events_taged['country/region_y'])]
        print("Count number of countries without regions : ",region_events_taged[pd.isnull(region_events_taged['country/region_y'])].__len__())
        print('Merge Regions to events data frame')
        events_cleaned_df_tmp = events_cleaned.merge(region_events_taged[['ISO3','country/region_y','M49Code_y','parent_id']], left_on = 'ISO3', right_on = 'ISO3', how = 'left')
    else:
        events_cleaned_df_tmp = region_events_taged
    return(events_cleaned_df_tmp)


def get_regions_idx(region_idx_gvkeyx_dict= {   'World': ['from comp_global.g_idx_index a',"where indexid similar to '%%(WORLD|ALL)%%' and indexval like '%%ALL%%'"],
                            'Australia and New Zealand' : ['from comp_global.g_idx_index a',"where conm similar to '%%(Australia|New Zealand)%%'","and indexgeo similar to  '%%(AUS|NZL)%%'", "and indextype in ('REGIONAL','REGSEC')"],
                            'Caribbean': ['from comp_global.g_idx_index a',"where conm similar to '%%(Latin America)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Central America': ['from comp_global.g_idx_index a',"where conm similar to '%%(Latin America)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Central Asia' : ['from comp_global.g_idx_index a',"where  conm similar to '%%(Asia)%%'"],
                            'Channel Islands': ['from comp_global.g_idx_index a',"where conm similar to '%%(Europe)%%'","and conm not similar to '%%(Emerging|Eastern|Ex |Excluding|Pacific|Australasia)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Eastern Africa' : ['from comp_global.g_idx_index a', "where conm similar to '%%(Africa)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Eastern Asia' : ['from comp_global.g_idx_index a',"where conm similar to '%%(Asia)%%'", "and not conm similar to '%%(Southeast)%%'", "and indextype in ('REGIONAL','REGSEC')","and indexid not in ('ISLAMIC')"],
                            'Eastern Europe': ['from comp_global.g_idx_index a',"where conm similar to '%%(Eastern Europe)%%'","and indexid not in ('ISLAMIC')"],
                            'Melanesia': ['from comp_global.g_idx_index a', "where conm similar to '%%(sia)%%'", "and  conm not similar to '%%(Europe|Southeast)%%'","and indexid not in ('ISLAMIC')"],
                            'Middle Africa' : ['from comp_global.g_idx_index a', "where conm similar to '%%(Africa)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Northern Africa' : ['from comp_global.g_idx_index a', "where conm similar to '%%(Africa)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Northern America' : ['from comp_global.g_idx_index a',"where conm similar to '%%(North America)%%'","and conm not similar to '%%(Europe|Southeast)%%'","and indexid not in ('ISLAMIC')"],
                            'Northern Europe' : ['from comp_global.g_idx_index a',"where  conm similar to '%%(Europe)%%'","and conm not similar to '%%(Emerging|Eastern|Austral|Pacific|Excluding UK|Ex UK|Ex Eurozone)%%'","and indexid not in ('ISLAMIC')"],
                            'South America' : ['from comp_global.g_idx_index a',"where conm similar to '%%(Latin America)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'South-eastern Asia' : ['from comp_global.g_idx_index a', "where  conm similar to '%%(Southeast|Asia)%%'","and conm not similar to '%%(India Pakistan)%%'","and indexgeo = 'APA' and indexid in ('WORLD', 'ALL')"],
                            'Southern Asia' : ['from comp_global.g_idx_index a', "where  conm similar to '%%(Southeast|Asia)%%'","and conm not similar to '%%(India Pakistan)%%'","and indexgeo = 'APA' and indexid in ('WORLD', 'ALL')"],
                            'Southern Europe' : ['from comp_global.g_idx_index a',"where  conm similar to '%%(Europe |Ex UK| Excluding UK)%%'","and conm not similar to '%%(Eastern Europe|Austra|Pacific|Including UK)%%'","and indextype in ('REGIONAL','REGSEC')","and indexid not in ('ISLAMIC')"],
                            'Western Africa' : ['from comp_global.g_idx_index a', "where conm similar to '%%(Africa)%%'","and indextype in ('REGIONAL','REGSEC')"],
                            'Western Asia' : ['from comp_global.g_idx_index a', "where  conm similar to '%%(Middle East)%%'"],
                            'Western Europe' : ['from comp_global.g_idx_index a',"where  conm similar to '%%(Europe)%%'","and conm not similar to '%%(Eastern Europe|Pacific|East|Excluding UK|Ex UK|Ex Eurozone|Emerging)%%'","and indextype in ('REGIONAL','REGSEC')","and indexid not in ('ISLAMIC')"]
                            }, RunWrdsAPI = True, wrds_username = 'fr497', password = 'Fr056301'):
    qry_results = pd.DataFrame()
    if RunWrdsAPI:
        print("Initialize WRDS connection")
        import wrds
        db = wrds.Connection(wrds_username= wrds_username, password= password)
        #glb_wrds_indices = pd.DataFrame()
        print("Running WRDS API Query and retrieving data into Object")
        wrds_query_df = pd.DataFrame()
        print("Running Queries from Dictionary")
        for k,v in region_idx_gvkeyx_dict.items():
            print(k)
            query_tmp = "select '"+ k +"' as RegionCode, * " + " ".join(v)
            wrds_query_df = db.raw_sql(query_tmp)
            print(wrds_query_df.shape)
            print("Appending to Data Frame")
            qry_results = pd.concat([qry_results,wrds_query_df], ignore_index=True)
        print("Returning Dataframe")
        return(qry_results)






def get_sector_idx( sector_idx_dict = { 'comp_global.g_idx_index':"where indexval in ('BANK','FINANCEOTH','FINANCIAL','FINSRV','FINSVCS','GENFIN','INVEST','MFINSVCS','PORTFOLIO','PRIME','SPECFIN','OTRFIN')"
                    ,'comp_na_daily_all.idx_index': "where conm similar to '%%(Financial|Financl|Banks|Diversfd Financial|Financials|FINAL EXCH & DATA|Specializd Finc|Asst Mgmt&Cus Bk|nvt Bkg & Brkg|Captl Mkts|Finance|FINANCIALS)%%'"
                    } ,RunWrdsAPI = True, events_cleaned_df = None , wrds_username='fr497',password = 'Fr056301' ):
    qry_results = pd.DataFrame()
    if RunWrdsAPI:
        print("Initialize WRDS connection")
        import wrds
        db = wrds.Connection(wrds_username=wrds_username, password=password)
        # glb_wrds_indices = pd.DataFrame()
        print("Running WRDS API Query and retrieving data into Object")
        wrds_query_df = pd.DataFrame()
        print("Running Queries from Dictionary")
        for k, v in sector_idx_dict.items():
            print(k)
            query_tmp = "select *, '" + k + "' as source from " + k + " " + v
            wrds_query_df = db.raw_sql(query_tmp)
            print(wrds_query_df.shape)
            print("Appending to Data Frame")
            qry_results = pd.concat([qry_results, wrds_query_df], ignore_index=True)

        if events_cleaned_df is not None:
            tmp1 = set(qry_results.indexgeo.unique())
            tmp2 = set(events_cleaned_df_regions.ISO3.unique())
            print("Sector Indices collected match the following Country Codes:", tmp1.intersection(tmp2).__len__())
            print(tmp1.intersection(tmp2))
        else:
            print(qry_results.indexgeo.unique())
        return(qry_results)



#Use a Wrapper function to combine all the indices tagged with the appropriate types. Region/Country/Sector

def get_idx_names(events_cleaned, Region = True, Country = True, Sector = True, RunWrdsAPI  =True):
    print("Getting Index names from WRDS COMPUSTAT Database")
    print("Initializing Dataframe")
    idx_names_df = pd.DataFrame()
    if Region:
        print("Getting Region-Level Index Names.")
        region_idx_names_tmp = get_regions_idx(RunWrdsAPI = RunWrdsAPI)
        region_idx_names_tmp['qrycat'] = "RegionLevel"
        region_idx_names_tmp['source'] = 'UN_M49Codes'
        idx_names_df = pd.concat([idx_names_df,region_idx_names_tmp])
    if Country:
        print("Getting Country-Level Index Names")
        country_idx_names_tmp = get_CountryIndices(events_cleaned,RunWrdsAPI = RunWrdsAPI)
        idx_names_df = pd.concat([idx_names_df,country_idx_names_tmp])
    if Sector:
        print("Getting Sector-Level Index Names")
        sector_idx_names_tmp = get_sector_idx(events_cleaned_df = events_cleaned_df_regions,RunWrdsAPI = RunWrdsAPI)
        sector_idx_names_tmp['qrycat'] = "SectorLevel"
        idx_names_df = pd.concat([idx_names_df,sector_idx_names_tmp], ignore_index= True)

    #Cleanup of objects required.
    #qrycat, source, etc. Should we join the region codes?
    idx_names_df = idx_names_df.reset_index(drop = True)
    return(idx_names_df)


def get_idx_prices(world_idx_names, events_cleaned, RunWrdsAPI=True, region=True, country=True, sector=True,
                   interval_before_mindate='2 year', interval_after_maxdate='2 year',
                   table_dict={'comp_global.g_idx_index': 'comp_global_daily.g_idx_daily',
                               'comp_na_daily_all.idx_index': 'comp_na_daily_all.idx_daily',
                               'UN_M49Codes': 'comp_global_daily.g_idx_daily'}
                   , wrds_username="fr497", password="Fr056301"):
    qry_results = pd.DataFrame()
    qry_results_str = []
    if RunWrdsAPI:
        print("Initialize WRDS connection")
        import wrds
        db = wrds.Connection(wrds_username=wrds_username, password=password)
    if sector:
        # Initialize the Wrds Connection

        print("Get date ranges for each Country in Sector File and Events file to get proper index data range")
        gb = events_cleaned[events_cleaned['ISO3'].isin(
            world_idx_names['indexgeo'][world_idx_names['qrycat'] == 'SectorLevel'].unique())]
        print("Getting Minimum and Maximum dates needed for each index from events object based on Country")
        gb = events_cleaned.groupby(["country", "ISO3"])
        # Get min and max and count of the date and events
        gb = gb.agg({'date': ['min', 'max', 'count']}).reset_index()
        # Sort the values by counts.
        gb = gb.ix[gb['date']['count'].sort_values(ascending=False).index].reset_index(drop=True)

        # prepare to query for each index based on source, gvkey, min/max date and append to a dataframe.

        qry_results_str = []
        for i in range(gb.__len__()):
            # temp
            # i = 0
            print("Getting Index Prices for Country Sector: " + gb['country'][i] + " | " + gb['ISO3'][i])

            gvkey_tmp = "','".join(world_idx_names['gvkeyx'][world_idx_names['indexgeo'] == gb['ISO3'][i]])
            source_tmp = world_idx_names['source'][world_idx_names['indexgeo'] == gb['ISO3'][i]].unique()
            print(source_tmp)

            print("Getting Appropriate Tables to get prices from")
            from_tmp_ls = []
            for k, v in table_dict.items():
                for j in source_tmp:
                    # print(j)
                    if k == j:
                        tmp = j.replace(k, v)
                        print(j, tmp)
                        from_tmp_ls.append(tmp)
            # print(newlist)

            print(from_tmp_ls)
            for y in range(from_tmp_ls.__len__()):
                print(from_tmp_ls[y])
                print("Creating Query String")
                select_tmp = "select * , '" + gb['ISO3'][i] + "' as indexgeo_mnt, '" + from_tmp_ls[y] + "' as table_src"
                from_tmp = "from " + from_tmp_ls[y]
                where_tmp = "where gvkeyx in ('" + gvkey_tmp + "') " \
                                                               " and datadate between date '" + gb['date']['min'][
                                i] + "' - INTERVAL '" + interval_before_mindate + "'" \
                                                                                  " and date '" + gb['date']['max'][
                                i] + "' + INTERVAL '" + interval_after_maxdate + "';"
                query_tmp = " ".join([select_tmp, from_tmp, where_tmp])
                # Run Query
                if RunWrdsAPI:
                    print("Querying Wrds")
                    qry_tmp_results = db.raw_sql(query_tmp)
                    if qry_tmp_results.empty:
                        print(query_tmp)
                    print("Appending to Results Dataframe")
                    qry_results = pd.concat([qry_results, qry_tmp_results])
                    print(qry_results.shape)

                else:
                    print("Appending query string to list")
                    qry_results_str.append(query_tmp)

    if country:

        print("Get date ranges for each COUNTRY in events file to get proper index data range")

        # Get data ranges for countries and ISO3 indices from the events
        # fill the nan values
        # events_cleaned['ISO3'][pd.isnull(events_cleaned["ISO3"])] = "N/A"
        # Aggregate on country and ISO3
        print("Getting Minimum and Maximum dates needed for each index from events object based on Country")
        gb = events_cleaned.groupby(["country", "ISO3"])
        # Get min and max and count of the date and events
        gb = gb.agg({'date': ['min', 'max', 'count']}).reset_index()
        # Sort the values by counts.
        gb = gb.ix[gb['date']['count'].sort_values(ascending=False).index].reset_index(drop=True)

        # prepare to query for each index based on source, gvkey, min/max date and append to a dataframe.

        qry_results_str = []
        for i in range(gb.__len__()):
            # temp
            # i = 0
            print("Getting Index Prices for Country: " + gb['country'][i] + " | " + gb['ISO3'][i])

            gvkey_tmp = "','".join(world_idx_names['gvkeyx'][world_idx_names['indexgeo'] == gb['ISO3'][i]])
            source_tmp = world_idx_names['source'][world_idx_names['indexgeo'] == gb['ISO3'][i]].unique()
            print(source_tmp)

            print("Getting Appropriate Tables to get prices from")
            from_tmp_ls = []
            for k, v in table_dict.items():
                for j in source_tmp:
                    # print(j)
                    if k == j:
                        tmp = j.replace(k, v)
                        print(j, tmp)
                        from_tmp_ls.append(tmp)
            # print(newlist)

            print(from_tmp_ls)
            for y in range(from_tmp_ls.__len__()):
                print(from_tmp_ls[y])
                print("Creating Query String")
                select_tmp = "select * , '" + gb['ISO3'][i] + "' as indexgeo_mnt, '" + from_tmp_ls[y] + "' as table_src"
                from_tmp = "from " + from_tmp_ls[y]
                where_tmp = "where gvkeyx in ('" + gvkey_tmp + "') " \
                                                               " and datadate between date '" + gb['date']['min'][
                                i] + "' - INTERVAL '" + interval_before_mindate + "'" \
                                                                                  " and date '" + gb['date']['max'][
                                i] + "' + INTERVAL '" + interval_after_maxdate + "';"
                query_tmp = " ".join([select_tmp, from_tmp, where_tmp])
                # Run Query
                if RunWrdsAPI:
                    print("Querying Wrds")
                    qry_tmp_results = db.raw_sql(query_tmp)
                    if qry_tmp_results.empty:
                        print(query_tmp)
                    print("Appending to Results Dataframe")
                    qry_results = pd.concat([qry_results, qry_tmp_results])
                    print(qry_results.shape)

                else:
                    print("Appending query string to list")
                    qry_results_str.append(query_tmp)

    if region:
        print("Get date ranges for all Events to get Regions Min-Max")
        print("Getting Minimum and Maximum dates needed for all events")
        tmp_date_max = events_cleaned['date'].max()
        tmp_date_min = events_cleaned['date'].min()
        tmp_gvkeyx = list(world_idx_names['gvkeyx'][world_idx_names['qrycat'] == 'RegionLevel'].unique())
        tmp_source = world_idx_names['source'][world_idx_names['qrycat'] == 'RegionLevel'].unique()
        print(tmp_source)
        tmp_source = "".join(tmp_source)
        print(table_dict[tmp_source])

        qry_results_str = []
        print("Getting Index Prices for Regions")
        gvkey_tmp = "','".join(tmp_gvkeyx)
        source_tmp = table_dict[tmp_source]
        print(source_tmp)
        print("Creating Query String")
        select_tmp = "select * , 'RegionalLevel' as indexgeo_mnt, '" + source_tmp + "' as table_src"
        from_tmp = "from " + source_tmp
        where_tmp = "where gvkeyx in ('" + gvkey_tmp + "') " \
                                                       " and datadate between date '" + tmp_date_min + "' - INTERVAL '" + interval_before_mindate + "'" \
                                                                                                                                                    " and date '" + tmp_date_max + "' + INTERVAL '" + interval_after_maxdate + "';"
        query_tmp = " ".join([select_tmp, from_tmp, where_tmp])
        # Run Query
        if RunWrdsAPI:
            print("Querying Wrds")
            qry_tmp_results = db.raw_sql(query_tmp)
            if qry_tmp_results.empty:
                print(query_tmp)
            print("Appending to Results Dataframe")
            qry_results = pd.concat([qry_results, qry_tmp_results])
            print(qry_results.shape)
        else:
            print("Appending query string to list")
            qry_results_str.append(query_tmp)

    if RunWrdsAPI:
        print("Drop Duplicates")
        qry_results = qry_results.drop_duplicates(subset=['gvkeyx', 'conm', 'datadate'])
        print(qry_results.shape)
        return (qry_results)
    else:
        return (qry_results_str)


#Workspace



# Webs ETF Index incorporate Webs Index
# Fd 17 Countries for Equirty Benchmark
# iShares





#Insertition Point
#TODO: Review all methods and put into a Class Libary so a DataFrame can be called from the object.


#events = []
#Run WebScrapers

#Get Events from websites
    #events = getevents_data()
    #Export the raw event file
    #events.to_csv("events.csv",sep = ",")

#Get UN Country Codes.
#From UN Website
    #event_CountryCodes = get_CountryCodes()
    #event_CountryCodes.to_csv("event_CountryCodes.csv",sep = ",")





#Import events file
#events = pd.read_csv("events.csv")
#event_CountryCodes = pd.read_csv("event_CountryCodes.csv")
#events_cleaned = events_normalize_annctype(events)
#events_cleaned = DeAgg_Events_Union(events_cleaned = events_cleaned, CombineFrame = True)
#events_cleaned = normalize_events_CountryCodes_UN(events_cleaned,event_CountryCodes)

#Get World Regions and attach to the events.
#events_cleaned_df_regions = get_country_regions(events_cleaned)
#events_cleaned_df_regions.to_csv("events_regions_df.csv", sep = ",")

events_cleaned_df_regions = pd.read_csv("events_regions_df.csv")

#Get all the indices names
#event_index_names_df = get_idx_names(events_cleaned_df_regions)
#event_index_names_df.to_csv("event_index_names_df.csv", sep = ",")
event_index_names_df = pd.read_csv("event_index_names_df.csv")


#Get index prices monthly and daily.
#Daily
#event_idx_prices = get_idx_prices(event_index_names_df,events_cleaned_df_regions)
#event_idx_prices.to_csv("regional_country_sector_idx_prices.csv", sep = ",")
event_idx_prices = pd.read_csv("regional_country_sector_idx_prices.csv")



#Event Study









#Make Function that can subset properly by iterating over object until all conditions are applied

def Event_Obj_Subsetter(EST_Events_Raw, cols = ["source","annctype","regioncode_x","global_idx_tic","regional_idx_tic","ISO3","country_idx_tic","sector_idx_tic"] ):
    final_df = pd.DataFrame()

    print("Convert NANs to String nan")
    EST_Events_Raw = EST_Events_Raw.replace(np.nan, "nan", regex=True)

    if "Unnamed: 0" in  EST_Events_Raw.columns:
        print("Drop Index Column")
        EST_Events_Raw = EST_Events_Raw.drop("Unnamed: 0", axis = 1)

    print("Getting Unique Value combinations for each Column")
    cols_vals_df = EST_Events_Raw[cols].drop_duplicates()



    print("Iterate to apply all matching criteria for each")
    for row_num_idx in range(cols_vals_df.shape[0]):
        tmp_df_obj = pd.DataFrame()
        tmp_df_match = EST_Events_Raw
        print(list(cols_vals_df.iloc[row_num_idx,]))
        for colname_idx in range(cols_vals_df.shape[1]):
            tmp_colname = cols_vals_df.columns[colname_idx]
            tmp_match = cols_vals_df.iloc[row_num_idx,colname_idx]
            #print(tmp_colname,tmp_match)
            print("Iterating Through Matching Rules to Subset Tmp Object")
            tmp_df_match = tmp_df_match[tmp_df_match[tmp_colname].str.endswith(tmp_match)]

        print("Applying GroupingVar:", "_".join(list(cols_vals_df.iloc[row_num_idx],)))
        tmp_df_match["GroupingVar"] = "_".join(list(cols_vals_df.iloc[row_num_idx],))


        if tmp_df_match.shape[0] > 0 :
            print("Appending Subsetted DataFame to Final Frame")
            print(tmp_df_match.shape)
            final_df = pd.concat([final_df,tmp_df_match])
        print("Garbage Collection")
        gc.collect()
     return(final_df)



#Generate Request File Regional Events  Map to World Indices

def EST_Event_Generator(events_cleaned_df_regions,event_index_names_inprices_df):
    events_cleaned_df_regions["Event ID"]  = events_cleaned_df_regions.index

    print("Merging to get Regional Market Indices")
    Request_File = events_cleaned_df_regions[["Event ID","date","annctype","category","ISO3","country","country/region_y","source"]].merge(event_index_names_inprices_df[["conm", "tic","regioncode", "gvkeyx"]][event_index_names_inprices_df['qrycat'] == "RegionLevel"], left_on = "country/region_y", right_on = "regioncode" , how = "left")
    print("Dropping Columns")
    Request_File = Request_File.drop(["country/region_y"], axis=1)
    print("Renaming Columns")
    Request_File = Request_File.rename({"conm": "regional_idx", "tic": "regional_idx_tic", "gvkeyx": "regional_idx_gvkeyx"}, axis=1)
    print(Request_File.shape)


    print("Merging to get Global Market Indices")
    Request_File["Global_match"] = "World"
    Request_File = Request_File.merge(event_index_names_inprices_df[["conm", "tic", "regioncode", "gvkeyx"]][event_index_names_inprices_df['qrycat'] == "RegionLevel"],left_on="Global_match", right_on="regioncode", how="left")
    print("Dropping Columns")
    Request_File = Request_File.drop(["Global_match", "regioncode_y"], axis=1)
    print("Renaming Columns")
    Request_File = Request_File.rename({"conm": "global_idx", "tic": "global_idx_tic", "gvkeyx": "global_idx_gvkeyx"},axis=1)
    print(Request_File.shape)

    print("Merging Country Level Indicies")
    Request_File = Request_File.merge(event_index_names_inprices_df[["conm", "tic", "indexgeo", "gvkeyx"]][~event_index_names_inprices_df["qrycat"].isin(["SectorLevel", "RegionLevel"])], left_on="ISO3", right_on="indexgeo",how="left")
    print("Dropping Columns")
    Request_File = Request_File.drop(["indexgeo"], axis=1)
    print("Renaming Columns")
    Request_File = Request_File.rename({"conm": "country_idx", "tic": "country_idx_tic", "gvkeyx": "country_idx_gvkeyx"}, axis=1)
    print(Request_File.shape)

    print("Merging Sector Level Indicies")
    Request_File = Request_File.merge(event_index_names_inprices_df[["conm", "tic", "indexgeo", "gvkeyx"]][event_index_names_inprices_df["qrycat"].isin(["SectorLevel"])],left_on="ISO3", right_on="indexgeo", how="left")
    print("Dropping Columns")
    Request_File = Request_File.drop(["indexgeo"], axis=1)
    print("Renaming Columns")
    Request_File = Request_File.rename({"conm": "sector_idx", "tic": "sector_idx_tic", "gvkeyx": "sector_idx_gvkeyx"},axis=1)


    print(Request_File.shape)
    return(Request_File)


#instead of loops, maybe merges are more appropirate.

#EST_Events_Raw = EST_Event_Generator(events_cleaned_df_regions,event_index_names_inprices_df)

#EST_Events_Raw.to_csv("EST_Events_Raw.csv", sep = ",")

EST_Events_Raw = pd.read_csv("EST_Events_Raw.csv")

#Correct Data Structures






def EST_File_Generator(EST_Events_Raw, event_idx_prices, event_index_names_df,
                        EstSubset = { "subset_columns" : [["annctype","global_idx","regional_idx","regioncode_x"]],
                                        "firmID": "regional_idx",
                                      "marketID":"global_idx"
                                    },
                        EstParams = {"start_ev_win": [-10],
                                     "end_ev_win" : [5],
                                     "end_est_win" : [-11],
                                     "est_win_len" : [120]} ):


    print("Convert NANs to String nan")
    EST_Events_Raw = EST_Events_Raw.replace(np.nan, "nan", regex=True)

    print("Setting Param Vector dataframe")
    tmp_param_df = pd.DataFrame.from_dict(EstParams)[list(EstParams.keys())]
    print("Setting Event Subset Rule")
    tmp_subset_df = pd.DataFrame.from_dict(EstSubset)

    tmp_rules = pd.concat([tmp_param_df,tmp_subset_df], axis = 1)

    print("Initializng Request File")
    Request_File_df = pd.DataFrame()
    print("Subsetting and creating Request File based on Rules")
    for row in range(tmp_rules.shape[0]):

        print("Getting Unique Value combinations for each Column")
        tmp_evt_subset_obj = EST_Events_Raw.drop_duplicates(subset = tmp_rules.iloc[row,tmp_rules.columns.get_loc("subset_columns")])


        print("Generating Grouping Variable and attached Params")
        tmp_evt_subset_obj["GroupingVar"] = tmp_evt_subset_obj[tmp_rules.iloc[row,tmp_rules.columns.get_loc("subset_columns")]].apply(lambda x : "_".join(x), axis = 1)
        tmp_evt_subset_obj["GroupingType"] = "_+_".join(tmp_rules.iloc[row,tmp_rules.columns.get_loc("subset_columns")])

        print("Dimensions of Object")
        print(tmp_evt_subset_obj.shape)


        print("Applying Params")
        for param in list(EstParams.keys()):
            tmp_evt_subset_obj[param]= tmp_rules.loc[row,param]

        print("Applying EST Request File formatting")

        request_headers = ["Event ID", EstSubset["firmID"],EstSubset["marketID"],"date","GroupingVar","GroupingType"] + (list(EstParams.keys()))



        tmp_evt_subset_obj = tmp_evt_subset_obj[request_headers]

        tmp_evt_subset_obj['date'] = pd.to_datetime(pd.to_datetime(tmp_evt_subset_obj['date']).dt.date).dt.strftime(
            "%d.%m.%Y")
        print("Sorting Results")
        tmp_evt_subset_obj = tmp_evt_subset_obj.sort_values(["date", "Event ID","GroupingVar", EstSubset["firmID"], EstSubset["marketID"]])

        print("Renaming Columns")
        tmp_evt_subset_obj = tmp_evt_subset_obj.rename({ EstSubset["firmID"] : 'firmID', EstSubset["marketID"] : "marketID"}, axis = 1)

        print("Removing Exact Matches with Firm and Market IDs or Nans in Firm ID")
        tmp_evt_subset_obj = tmp_evt_subset_obj[tmp_evt_subset_obj["firmID"] != tmp_evt_subset_obj["marketID"]]

        tmp_evt_subset_obj = tmp_evt_subset_obj[tmp_evt_subset_obj["firmID"] != "nan"]


        print("Appending Subset to Request DataFrame")
        Request_File_df = pd.concat([Request_File_df,tmp_evt_subset_obj])


    print("Applying Sequential Numbering for Event ID duplicate but different Firm and Market Streams")
    Request_File_df["dup_stream_num"] = Request_File_df.sort_values(['Event ID','firmID','marketID']).groupby(['Event ID']).cumcount() + 1

    Request_File_df = Request_File_df.sort_values(["Event ID", "firmID","dup_stream_num"]).reset_index(drop=True)
    print("Completed Request_File")

    print("Filter event_index _names to those that exist in event_idx_prices")
    event_index_names_inprices_df = event_index_names_df[(event_index_names_df["gvkeyx"].isin(event_idx_prices["gvkeyx"].astype(int).unique())) & (event_index_names_df["indexid"] != "ISLAMIC")]
    event_idx_prices_in_idx = event_idx_prices.merge(event_index_names_inprices_df, on="gvkeyx", how='left')


    print("Generating Firm Price Data File from Wrds Prices File")
    FirmData_File_df = event_idx_prices_in_idx[["conm", "datadate", "prccd"]][event_idx_prices_in_idx["conm"].isin(Request_File_df["firmID"].unique())].sort_values(["datadate", "conm"]).drop_duplicates().reset_index(drop=True)
    FirmData_File_df['datadate'] = pd.to_datetime(pd.to_datetime(FirmData_File_df['datadate']).dt.date).dt.strftime("%d.%m.%Y")
    FirmData_File_df = FirmData_File_df.rename({'conm' : "firmID",'datadate':'date','prccd':'price'})

    print("Generating Market Price Data File from Wrds Prices File")
    MarketData_df = event_idx_prices_in_idx[["conm", "datadate", "prccd"]][event_idx_prices_in_idx["conm"].isin(Request_File_df["marketID"].unique())].sort_values(["datadate", "conm"]).drop_duplicates().reset_index(drop=True)
    MarketData_df['datadate'] = pd.to_datetime(pd.to_datetime(MarketData_df['datadate']).dt.date).dt.strftime("%d.%m.%Y")
    MarketData_df = MarketData_df.rename({'conm': "firmID", 'datadate': 'date', 'prccd': 'price'})



    return{'RequestData': Request_File_df, 'FirmData': FirmData_File_df,'MarketData': MarketData_df}


#Need to number the Duplicated Event IDs ******DONE
#Need to add the data for subsetting **********DONE
#Drop where FirmID and Market ID are the same **************DONE
#may have to consider the Nan value variations *********** DONE
#Generate the Firm Data and Market Data files as well as Request File.************** DONE


#Pass through to API in R
#Retrieve Results and combine into Dataset for analysis



PreProcess_EST_Data_Dict   = EST_File_Generator(EST_Events_Raw,event_idx_prices,event_index_names_df)



test = EST_R_API_Wrapper(PreProcess_EST_Data_Dict)

def EST_R_API_Wrapper(PreProcess_EST_Data_Dict,
                      params_dict = {
                         'workingdir' : '.',
                         'apiKey' : '573e58c665fcc08cc6e5a660beaad0cb',
                          'apiUrl' :  "http://api.eventstudytools.com",
                          'ResultFileType':'csv',
                          'ReturnType':'log',
                          'NonTradingDays':'earlier',
                          'BenchmarkModel':'mm',
                          'resultPath' : './results/',
                          'requestFile': '01_RequestFile_df.csv',
                          'firmDataFile': '02_FirmData_df.csv',
                          'marketDataFile': '03_MarketData_df.csv' }):

    print("Starting EST API Steps")

    print("Setting Working Directory")
    workdir = os.chdir(params_dict['workingdir'])
    print(os.getcwd())


    print("Generating CSV Files for R API EST")

    #May have to loop through he duplicates.
    #PreProcess_EST_Data_Dict_bk = PreProcess_EST_Data_Dict
    # PreProcess_EST_Data_Dict = PreProcess_EST_Data_Dict_bk



    print("Initializing Results Objects")
    estAnalysisReport  = pd.DataFrame()
    estARresults = pd.DataFrame()
    estAARresults = pd.DataFrame()
    estCARresults = pd.DataFrame()
    estCAARresults = pd.DataFrame()

    for dup_num in range(1,PreProcess_EST_Data_Dict["RequestData"]["dup_stream_num"].max()):
        print("dup_num_stream:", dup_num )
        PreProcess_EST_Data_Dict["RequestData"] =  PreProcess_EST_Data_Dict["RequestData"][PreProcess_EST_Data_Dict["RequestData"]["dup_stream_num"] == dup_num][["Event ID", "firmID","marketID", "date","GroupingVar","start_ev_win","end_ev_win","end_est_win","est_win_len"]]
        PreProcess_EST_Data_Dict["RequestData"]["GroupingVar"] = PreProcess_EST_Data_Dict["RequestData"]["GroupingVar"].apply(lambda x : x + "_" + str(dup_num))


        print("Creating CSV files to pass to R API")
        PreProcess_EST_Data_Dict['RequestData'].to_csv(params_dict["requestFile"], sep=';', header=False, index=False, line_terminator="\n")
        PreProcess_EST_Data_Dict['FirmData'][PreProcess_EST_Data_Dict['FirmData']['conm'].isin(list(PreProcess_EST_Data_Dict['RequestData']['firmID'].unique()))].to_csv(params_dict["firmDataFile"], sep=';', header=False, index=False, line_terminator="\n")
        PreProcess_EST_Data_Dict['MarketData'][PreProcess_EST_Data_Dict['MarketData']['conm'].isin(list(PreProcess_EST_Data_Dict['RequestData']['marketID'].unique()))].to_csv(params_dict["marketDataFile"], sep=';', header=False, index=False, line_terminator="\n")



        est_api_wrapper = ro.r('''
            
       
            
            if (!require("EventStudy")) {
              if (!require("devtools")) {
              install.packages("devtools")
                }
                devtools::install_github("EventStudyTools/api-wrapper.r")
            }
            
            library(EventStudy)
            
            setwd("'''+ os.getcwd() +'''")
            
            apiUrl <- "'''+ params_dict["apiUrl"] +'''"
            
            apiKey <- "'''+ params_dict["apiKey"] + '''"
            
            
            estAPIKey(apiKey)
            estSetup <- EventStudyAPI$new(apiUrl)
            
            arcParams <- ARCApplicationInput$new()
            arcParams$setResultFileType("'''+ params_dict["ResultFileType"] + '''")
            arcParams$setReturnType("'''+ params_dict["ReturnType"] + '''")
            arcParams$setNonTradingDays("'''+ params_dict["NonTradingDays"] + '''")
            arcParams$setBenchmarkModel("'''+ params_dict["BenchmarkModel"] + '''")    
            
            #mm (default): Market Model
            #mm-sw: Scholes/Williams Model
            #cpmam: Comparison Period Mean Adjusted
            #ff3fm: Fama-French 3 Factor Model
            #ffm4fm: Fama-French-Momentum 4 Factor Model
            #garch: GARCH (1, 1) Model
            #egarch: EGARCH (1, 1) Model
    
            estSetup$authentication(apiKey)
            estSetup$performEventStudy(estParams = arcParams,
                               dataFiles = c("request_file" = "'''+ params_dict["requestFile"] + '''",
                                             "firm_data" = "'''+ params_dict["firmDataFile"] + '''",
                                             "market_data" = "'''+ params_dict["marketDataFile"] + '''"),
                               downloadFiles = T)
                        
    ''')
        print("Reading in Results from API to PY DataFrame")
        estAnalysisReport_tmp = pd.read_csv("./results/analysis_report.csv", sep = ';')
        estARresults_tmp = pd.read_csv("./results/ar_results.csv", sep=';')
        estAARresults_tmp = pd.read_csv("./results/aar_results.csv", sep = ';')
        estCARresults_tmp = pd.read_csv("./results/car_results.csv", sep = ';')
        estCAARresults_tmp = pd.read_csv("./results/caar_results.csv", sep = ';')

        print("Appending to Results Object")
        estAnalysisReport =  pd.concat([estAnalysisReport, estAnalysisReport_tmp])
        estARresults = pd.concat([estARresults, estARresults_tmp])
        estAARresults = pd.concat([estAARresults, estAARresults_tmp])
        estCARresults = pd.concat([estCARresults, estCARresults_tmp])
        estCAARresults = pd.concat([estCAARresults,estCAARresults_tmp])






    return{'AnalysisReport':estAnalysisReport, 'AR_Results': estARresults, 'AAR_Results': estAARresults, 'CAR_Results': estCARresults, "CAAR_Results": estCAARresults}







#Cluster Analysis of Returns around Events.




#Participant Level
#Will need to make a list of banks in US, Euro, and Asia that are relevant and get their market data.













