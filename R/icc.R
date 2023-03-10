library(readr)
library(irr)

feature_1 <- read_csv('./features1.csv')
feature_2 <- read_csv('./features2.csv')
len <- 1068
icc_val<-vector(length=len)
thr <- 0.9

selected <- feature_1
for (i in 2:len){
  ratings <- cbind(selected[,i],feature_2[,i])
  icc <- icc(ratings, model = "twoway", 
             type = "agreement", 
             unit = "single", r0 = 0, conf.level = 0.95)
  icc_val[i-1] <- icc$value
}
Index <- which(icc_val > thr)
dim(icc_val)=c(1,len)
write.csv( icc_val,file = './output.csv',row.names = F ) 
