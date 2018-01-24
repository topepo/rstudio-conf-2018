###################################################################
## Code for Applied Machine Learning by Max Kuhn @ RStudio::conf
## https://github.com/topepo/rstudio-conf-2018

library(ggplot2)

thm <- theme_bw() + 
  theme(
    panel.background = element_rect(fill = "transparent", colour = NA), 
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.position = "top",
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.key = element_rect(fill = "transparent", colour = NA)
  )
theme_set(thm)

# Slide 10

library(AmesHousing)
ames <- make_ames()
nrow(ames)

library(rsample)

# Make sure that you get the same random numbers
set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price")

ames_train <- training(data_split)
ames_test  <- testing(data_split)

nrow(ames_train)/nrow(ames)

# Slide 11

ggplot(ames_train, aes(x = Sale_Price)) + 
  geom_line(stat = "density", trim = TRUE) + 
  geom_line(data = ames_test, 
            stat = "density", 
            trim = TRUE, col = "red") 

# Slide 15

simple_lm <- lm(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)

library(broom)
simple_lm_values <- augment(simple_lm)
names(simple_lm_values)

# Slide 19

summary(simple_lm)

# Slide 28

library(rsample)
set.seed(2453)
cv_splits <- vfold_cv(ames_train, v = 10, strata = "Sale_Price")
cv_splits

# Slide 29

# The `split` objects contain the information about the sample sizes
cv_splits$splits[[1]]

# Use the `analysis` and `assessment` functions to get the data
analysis(cv_splits$splits[[1]]) %>% dim()
assessment(cv_splits$splits[[1]]) %>% dim()

# Slide 30

library(yardstick)
library(dplyr)

lm_fit <- function(data_split, ...) 
  lm(..., data = analysis(data_split))

# A formula is also needed for each model:
form <- as.formula(
	log10(Sale_Price) ~ Longitude + Latitude
	)

model_perf <- function(data_split, mod_obj) {
  vars <- rsample::form_pred(mod_obj$terms)
  assess_dat <- assessment(data_split) %>%
  	select(!!!vars, Sale_Price) %>%
  	mutate(
  		pred = predict(
  			mod_obj, 
  			newdata = assessment(data_split)
  		),
  		Sale_Price = log10(Sale_Price)
  	)
  
  rmse <- assess_dat %>% 
  	rmse(truth = Sale_Price, estimate = pred)
  rsq <- assess_dat %>% 
  	rsq(truth = Sale_Price, estimate = pred)
  data.frame(rmse = rmse, rsq = rsq)
}

# Slide 31

library(purrr)
cv_splits <- cv_splits %>%
  mutate(lm_mod = map(splits, lm_fit, formula = form))
cv_splits

# Slide 32

# map2 can be used to move over two objects of equal length
lm_res <- map2_df(cv_splits$splits, cv_splits$lm_mod, model_perf) %>% 
  dplyr::rename(rmse_simple = rmse, rsq_simple = rsq)
head(lm_res, 3)

## Merge in results:
cv_splits <- cv_splits %>% bind_cols(lm_res)

## Rename the columns and compute the resampling estimates:
cv_splits %>% select(rmse_simple, rsq_simple) %>% colMeans

# Slide 34

get_assessment <- function(splits, model) 
  augment(model, newdata = assessment(splits)) %>%
  	mutate(.resid = log10(Sale_Price) - .fitted)

holdout_results <- map2_df(cv_splits$splits, cv_splits$lm_mod, get_assessment)
holdout_results %>% dim()
ames_train %>% dim()

# Slide 39

library(caret)

knn_train_mod <- knnreg(log10(Sale_Price) ~ Longitude + Latitude, 
                        data = ames_train,
                        k = 2)
repredict <- data.frame(price = log10(ames_train$Sale_Price)) %>%
	mutate(pred = 
				 	predict(knn_train_mod, 
				 					newdata = ames_train %>% select(Longitude, Latitude)
				 	)
	)

repredict %>% rsq(truth = "price", estimate = "pred") # <- the ruckus is here

# Slide 40

knn_fit <- function(data_split, ...) 
  knnreg(..., data = analysis(data_split))

cv_splits <- cv_splits %>%
	mutate(knn_mod = map(splits, knn_fit, formula = form, k = 2))

knn_res <- map2_df(cv_splits$splits, cv_splits$knn_mod, model_perf) %>% 
  rename(rmse_knn = rmse, rsq_knn = rsq)

## Merge in results:
cv_splits <- cv_splits %>% bind_cols(knn_res)

colMeans(knn_res)

# Slide 42

rs_comp <- data.frame(
	rmse = c(cv_splits$rmse_simple, cv_splits$rmse_knn),
	Model = rep(c("Linear\nRegression", "2-NN"), each = nrow(cv_splits)),
	Resample = cv_splits$id
)

ggplot(rs_comp, aes(x = Model, y = rmse, group = Resample, col = Resample)) + 
  geom_point() + 
  geom_line() + 
  theme(legend.position = "none")

# Slide 43

t.test(cv_splits$rmse_simple, cv_splits$rmse_knn, paired = TRUE)

# Slide 49

knn_rmse <- function(k, split) {
	mod <- knnreg(log10(Sale_Price) ~ Longitude + Latitude, 
								data = analysis(split),  
								k = k)
	# Extract the names of the predictors
	preds <- form_pred(mod$terms)
	data.frame(Sale_Price = log10(assessment(split)$Sale_Price)) %>%
		mutate(pred = predict(mod, assessment(split) %>% select(!!!preds))) %>%
		rmse(Sale_Price, pred)
}

# Slide 50
library(dplyr)

knn_grid <- function(split) {
	# Create grid
  tibble(k = 1:20) %>%
    # Execute grid for this resample
    mutate(
      rmse = map_dbl(k, knn_rmse, split = split),
      # Attach the resample indicators using `lables`
      id = labels(split)[[1]]
    )
}

# Slide 51

iter_over_resamples <- 
	function(resamp) 
		map_df(resamp$splits, knn_grid)

# Slide 52

knn_tune_res <- iter_over_resamples(cv_splits)
knn_tune_res %>% head(15)

# Slide 53

rmse_by_k <- knn_tune_res %>%
  group_by(k) %>%
  summarize(rmse = mean(rmse))
            
ggplot(rmse_by_k, aes(x = k, y = rmse)) + 
  geom_point() + geom_line()

# Slide 54

best_k <- knn_tune_res %>%
  group_by(id) %>%
  summarize(k = k[which.min(rmse)],
            rmse = rmse[which.min(rmse)])

ggplot(rmse_by_k, aes(x = k, y = rmse)) + 
  geom_point() + geom_line() + 
  geom_line(data = knn_tune_res, 
            aes(group = id, 
                col = id),
            alpha = .2, lwd = 1) + 
  geom_point(data = best_k, aes(col = id),
             alpha = .5, cex = 2)+
  theme(legend.position = "none")

# Slide 55

final_knn <- knnreg(log10(Sale_Price) ~ Longitude + Latitude,
                    data = ames_train,
                    k = 4)

