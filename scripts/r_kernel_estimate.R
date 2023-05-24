#install.packages("kdensity") #Uncomment if R package not on system.
library("kdensity")
kestimate <- function(data_arr,nums){
  kde <- density(data_arr ,kernel = "gaussian")
  kern.samp <- sample(kde$x,nums,replace=TRUE,prob=kde$y)
}
