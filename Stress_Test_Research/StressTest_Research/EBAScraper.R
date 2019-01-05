tryCatch(
  url %>%
    as.character() %>% 
    read_html() %>% 
    html_nodes('h1') %>% 
    html_text(), 
  error = function(e){NA}    # a function that returns NA regardless of what it's passed
)



if (!require("rvest")) {
  install.packages("rvest")
}
library("rvest")


EBA_StressTest = data.frame()

urlbase = "https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-stress-testing"

#ebayears = c(2009,2010,2011,2014,2016,2018)

ebayears = 2000:2019

for (i in ebayears){

  tmpurl = paste0(urlbase,"/",i)
  print(tmpurl)
  
  if(!is.na(tryCatch( tmpurl %>% as.character() %>% read_html(), error = function(e){NA})))
     {
  
      webpage = read_html(tmpurl)
      #scraptmp = html_nodes(webpage, "dl")
      
      #Short Title
      title = webpage %>% html_nodes('dl') %>% html_node("a") %>% html_text()
      #Description
      desc = webpage %>% html_nodes('dl') %>% html_node("dd") %>% html_text()
      #Date
      date = as.Date(webpage %>% html_nodes('dl') %>% html_node("dd.TLDate") %>% html_text(), format = "%d/%m/%Y")
      #URL
      url = paste0("https://www.eba.europa.eu", webpage %>% html_nodes('dl') %>% html_node("a") %>% html_attr("href"))
      
      tmpobj = data.frame(title,desc,date,url) 
      
      EBA_StressTest = rbind.data.frame(EBA_StressTest, tmpobj)
  }
}

dim(EBA_StressTest)
View(EBA_StressTest)
