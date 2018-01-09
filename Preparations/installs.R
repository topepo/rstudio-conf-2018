from_cran <- 
	c("AmesHousing", "broom", "caret", "devtools", "doParallel", "e1071", "earth", 
		"glmnet", "ipred", "klaR", "pROC", "rpart", "rsample", "sessioninfo", 
		"tidyposterior", "tidyverse", "yardstick")

install.packages(from_cran, repos = "http://cran.rstudio.com")

library(devtools)

install_github("topepo/recipes")

# If you can install from source: 
install_github("topepo/caret", subdir = "pkg/caret")

if(!interactive())
  q("no")
