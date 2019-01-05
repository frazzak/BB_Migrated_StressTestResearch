library("devtools")
library(XML)
library("feedeR")
library(dplyr)
library(stringr)

yearrange = 2000:2019
policyareas = c("non-bank-financial-intermediation", "vulnerabilities-assessments")




rss_df_1 = data.frame()
ytmp = c()

for (y in policyareas){

  print(paste0("Policy Area: ",y))
  #Policy Areas List
  ytmp = c(ytmp,y)
  
  #Tmp rss iteration for policy area
  rss_df_tmp = data.frame()
  
  for (i in (yearrange))
      {
    url =paste0("http://www.fsb.org/",i,"/feed/?policy_area=",y)
    urlrss = feed.extract(url)
    #print(urlrss$items)
    
    if(nrow(urlrss$items) > 0) {
      names(urlrss$items)[4] = "PolicyArea"
      urlrss$items[,4] = y
      rss_df_tmp = rbind(rss_df_tmp,urlrss$items[,1:4])
    }
  
    
      }
 #Dup Checking and tagging for removal
 rss_df_1 = rbind(rss_df_1,rss_df_tmp)
 #dim(rss_df_1)
 dupidx = duplicated(rss_df_1[,1:3])
 rss_df_1[dupidx,4] = str_c(ytmp, collapse = ", ")
 
}

#Dup Removal
rss_df_1 <-rss_df_1 %>% group_by(title,date) %>% filter(duplicated(title,date) | n()==1)




