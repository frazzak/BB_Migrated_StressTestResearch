
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

    #B. Region Level Index Returns
        #Ticker, GVKEY, PERMCO, PERMNO
    #C. Sector Level Index Returns
     #Ticker, GVKEY, PERMCO, PERMNO
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


#TODO: Can improve this function using Python Dictionary to tag based on source, announcement type, title, keywords
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



#Make into function.
#Takes tagged events with UN_CountryCodes
#Runs Rules to fix country names to fit UN Code standard.




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



#JSON to Pandas Dataframe.
#UN Country Data
# Reporting Countries and Areas
def get_CountryCodes(excelurl = "http://unstats.un.org/unsd/tradekb/Attachment440.aspx?AttachmentType=1"
                    ,jsonurl = ["https://comtrade.un.org/data/cache/reporterAreas.json","https://comtrade.un.org/data/cache/partnerAreas.json"]
                    ,weburl = ["https://www.nationsonline.org/oneworld/country_code_list.htm"]):

    print("Initializing Dataframe for CountryCodes")
    CountryCodes = pd.DataFrame()

    if len(excelurl) > 0:
        for url in excelurl:
            print("Getting Excel File from: "+ url)
            CountryCodes_tmp = pd.read_excel(url)

            CountryCodes_tmp = CountryCodes_tmp["Country Code", "Country Name, Abbreviation", "ISO2-digit Alpha", "ISO3-digit Alpha"]
            CountryCodes_tmp.columns = ["UN_CountryCode", "CountryName", "ISO2", "ISO3"]
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
            CountryCodes_tmp.columns = ["UN_CountryCode", "CountryName", "ISO2", "ISO3"]
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
            CountryCodes_tmp.columns = ["UN_CountryCode", "CountryName", "ISO2", "ISO3"]
            print("Adding Rows to Initialized Dataframe.")
            #UN Dataframe combined
            CountryCodes = pd.concat([CountryCodes,CountryCodes_tmp],ignore_index= True)

        print("Dropping Duplicates")
        CountryCodes = CountryCodes.drop_duplicates()


        print("Manually Adding Countries that are Missing")
        #Added Kosovo, Gurensey,Jersey

        print("Adding Kosovo")
        tmp = {"id":"XKX","text":"Kosovo"}
        CountryCodes = UN_CountryCodes.append(tmp, ignore_index = True)


        print("Dropping Duplicates after Manual Add")
        CountryCodes = UN_CountryCodes.drop_duplicates()

        print("Resetting Index")
        CountryCodes = UN_CountryCodes.reset_index()

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
                                                                                columns_orig = ["title","date","country","category","source","annctype","BankingUnion"]):
    print("Creating Rule Set")
    normalize_dict_pd = pd.DataFrame.from_dict(findstr_dict)
    normalize_dict_pd = normalize_dict_pd.sort_values("ruleorder")
#    print("Sotring Original Columns")

    print("Initial Join to UN_CountryCodes")
    events_cleaned = events_cleaned.merge(UN_CountryCodes,left_on = 'country', right_on = 'text', how = 'left' )

    print("Applying Rules")
    for i in range(len(normalize_dict_pd)):
        if normalize_dict_pd.loc[i]["rule"] == 1:
            print("RuleOrder: " + str(normalize_dict_pd.loc[i]["ruleorder"])  + " Rule: " + str(normalize_dict_pd.loc[i]["rule"]) + " Source: " + normalize_dict_pd.loc[i]["source"])
            print(" Replace Country Title: " + normalize_dict_pd.loc[i]["findstr"] + " with: " + normalize_dict_pd.loc[i]["replacestr"])
            #Get Index Rows for matching Pattern
            idxrows = events_cleaned["country"][(pd.isnull(events_cleaned["text"]))
                                                & (events_cleaned["source"] == normalize_dict_pd.loc[i]["source"])
                                                & (events_cleaned["country"].str.startswith(normalize_dict_pd.loc[i]["findstr"]))].index

            #Replace the String in subset
            events_cleaned["country"][idxrows] = events_cleaned["country"][idxrows].str.replace(normalize_dict_pd.loc[i]["findstr"], normalize_dict_pd.loc[i]["replacestr"])


            #Merge to the UNcodes to see how many resolve for next iteration
            events_cleaned = events_cleaned[columns_orig].merge(UN_CountryCodes, left_on='country', right_on='text',
                                                            how='left')

        if normalize_dict_pd.loc[i]["rule"] == 2:
            print("RuleOrder: " + str(normalize_dict_pd.loc[i]["ruleorder"]) + " Rule: " + str(normalize_dict_pd.loc[i]["rule"]) + " Source: " + normalize_dict_pd.loc[i]["source"])
            print(" Replace Country Title: " + normalize_dict_pd.loc[i]["findstr"] + " with: " + normalize_dict_pd.loc[i]["replacestr"])

            events_cleaned["country"][(pd.isnull(events_cleaned['text'])) & (events_cleaned["source"] == normalize_dict_pd.loc[i]["source"]) & (
            events_cleaned["country"].str.contains(normalize_dict_pd.loc[i]["findstr"]))] = normalize_dict_pd.loc[i]["replacestr"]

    print("Final Merge with all Rules completed")
    events_cleaned = events_cleaned[columns_orig].merge(UN_CountryCodes, left_on='country', right_on='text',
                                                        how='left')
    print("Number of rows that did not match: ", events_cleaned["country"][pd.isnull(events_cleaned['text'])].__len__())



    print("Dropping Index Column")
    events_cleaned = events_cleaned.drop("index", axis =1)
    print('Renaming Columns')
    events_cleaned = events_cleaned.rename(columns = {"id" : "UN_CodeID","text":"UN_CountryName"})
    #print(events_cleaned.columns)
    #columns_orig.extend(["UNCode_id","UNCountry_Name"])
    #print(columns_orig)
    #events_cleaned.columns = columns_orig
    return(events_cleaned)




countrycodeurl = "https://www.nationsonline.org/oneworld/country_code_list.htm"



test = FICUNCodes.merge(UN_CountryCodes, left_on='UNCode_ID', right_on='id',
                                                        how='left').sort_values("id")



test2 = events_cleaned.merge(FICUNCodes, left_on = "country", right_on = "country", how  = "left")

test2 = test2.sort_values("UNCode_ID")

test2["country"][(pd.isnull(test2["UNCode_ID"]))].unique()

test2 = events_cleaned.merge(FICUNCodes, left_on = "country", right_on = "country", how  = "left")



UN_CountryCodes[UN_CountryCodes["text"].str.contains("USA")]

df[["Country Code","Country Name, Abbreviation","ISO2-digit Alpha","ISO3-digit Alpha"]]

#First Join with UN Codes
#Then run again for the Nans to join with the web scraped.
url = "http://unstats.un.org/unsd/tradekb/Attachment440.aspx?AttachmentType=1"
df = pd.read_excel(url)


test2 = events_cleaned.merge(df, left_on = "country", right_on = "Country Name, Abbreviation", how  = "left")

test2 = test2.sort_values("Country Name, Abbreviation")

test3 = test2[(pd.isnull(test2["BankingUnion"])) & (pd.isnull(test2["Country Name, Abbreviation"]))]

test4 = test3[["title","country"]].merge(FICUNCodes, left_on = "country", right_on = "country", how  = "left")


test3

#import requests
#resp = requests.get(url)
#output = open("test.xls",'wb')
#output.write(resp.content)
#output.close()





#DeAggUnionListUN = DeAgg_UnionList.merge(UN_CountryCodes,left_on = 'Country', right_on = 'text', how = 'left' )

#DeAggUnionListUN = DeAggUnionListUN.sort_values("text")
#No Nans in UN code. All Unions Match out.




#Can make this into a method?
#Deaggregate the Events
    #apply new countries
    #and remove the events with Unions.
    #Make new column to indicate country is part of a union.

#Create Union column
#Need to find way to pass multiple arguements, and lists for a iterable look to consolidate this logic.

#Then attach to FIC codes and UN Codes and Country Codes
#https://www.nationsonline.org/oneworld/country_code_list.htm
#Has Country name , Alpha 2 , Alpha 3, and UN CODE





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
UN_CountryCodes = get_UNCountryCodes()

#Get from Nations Online


#Import events file
events = pd.read_csv("events.csv")
events_cleaned = events_normalize_annctype(events)
events_cleaned = DeAgg_Events_Union(events_cleaned = events_cleaned, CombineFrame = True)
events_cleaned = normalize_events_CountryCodes_UN(events_cleaned,UN_CountryCodes)
