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

# Slide 4

load("Data/car_data.RData")
library(dplyr)
car_train <- car_data %>%
  filter(model_year < 2018)

car_test <- car_data %>%
  filter(model_year == 2018)

# Slide 8

## lm(mpg ~ . -carline + poly(eng_displ, 2), data = car_train)

# Slide  9

car_train %>%
  group_by(division) %>%
  count() %>%
  arrange(n) %>%
  head(8)

# Slide 10

library(recipes)
basic_rec <- recipe(mpg ~ ., data = car_train) %>%
  # keep the car name but don't use as a predictor
  add_role(carline, new_role = "car name") %>%
  # collapse some divisions into "other"
  step_other(division, threshold = 0.005) %>%
  step_dummy(all_nominal(), -carline) %>%
  step_zv(all_predictors())

# Slide 14

glmn_grid <- expand.grid(alpha = seq(0, 1, by = .25), lambda = 10^seq(-3, -1, length = 20))
nrow(glmn_grid)

# Slide 17 

library(caret)
ctrl <- trainControl(
  method = "cv", 
  # Save the assessment predictions from the best model
  savePredictions = "final",
  # Log the progress of the tuning process
  verboseIter = TRUE
  )

# Slide 18

glmn_rec <- basic_rec %>%
  step_poly(eng_displ) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

set.seed(3544)
glmn_mod <- train(
  glmn_rec, 
  data = car_train,
  method = "glmnet", 
  trControl = ctrl,
  tuneGrid = glmn_grid
  )

# Slide 20

ggplot(glmn_mod) + scale_x_log10() + theme(legend.position = "top")

# Slide 21

glmn_mod$pred %>% head(4)

ggplot(glmn_mod$pred, aes(x = obs, y = pred)) +
  geom_abline(col = "green", alpha = .5) + 
  geom_smooth(se = FALSE, col = "red", lty = 2, lwd = 1, alpha = .5) + 
  geom_point(alpha = .3)

# Slide 22

reg_imp <- varImp(glmn_mod, scale = FALSE)
ggplot(reg_imp, top = 30) + xlab("")

# Slide 24

library(glmnet)
plot(glmn_mod$finalModel, xvar = "lambda")

# If you would like to run in parallel

library(doParallel)
cl <- makeCluster(parallel::detectCores(logical = FALSE))
registerDoParallel(cl)

# Slide 37
  
ctrl$verboseIter <- FALSE

mars_grid <- expand.grid(degree = 1:2, nprune = seq(2, 60, by = 2))

# Using the same seed to obtain the same 
# resamples as the glmnet model.
set.seed(3544)
mars_mod <- train(
  basic_rec, 
  data = car_train,
  method = "earth",
  tuneGrid = mars_grid,
  trControl = ctrl
)

# Slide 40

ggplot(mars_mod) + theme(legend.position = "top")

# Slide 41

ggplot(mars_mod$pred, aes(x = obs, y = pred)) +
  geom_abline(col = "green", alpha = .5) + 
  geom_smooth(se = FALSE, col = "red", 
              lty = 2, lwd = 1, alpha = .5) + 
  geom_point(alpha = .3)

# Slide 42

library(earth)
mars_mod$finalModel

# Slide 43

mars_imp <- varImp(mars_mod)
ggplot(mars_imp, top = 30) + xlab("")

# Slide 44

set.seed(3544)
mars_gcv_mod <- train(
  basic_rec, 
  data = car_train,
  method = "gcvEarth",
  tuneGrid = data.frame(degree = 1:2),
  trControl = ctrl
)
mars_gcv_mod$finalModel

# Slide 49

set.seed(3544)
mars_gcv_bag <- train(
  basic_rec, 
  data = car_train,
  method = "bagEarthGCV",
  tuneGrid = data.frame(degree = 1:2),
  trControl = ctrl,
  # Number of bootstraps for `bagEarth` function
  B = 50
)

# Slide 50

mars_gcv_bag

# Slide 51

ggplot(mars_gcv_bag$pred, aes(x = obs, y = pred)) +
  geom_abline(col = "green", alpha = .5) + 
  geom_smooth(se = FALSE, col = "red", 
              lty = 2, lwd = 1, alpha = .5) + 
  geom_point(alpha = .3)

# Slide 52

car_train %>% 
  arrange(mpg) %>% 
  select(mpg, carline, model_year) %>% 
  tail(10)

# Slide 54

rs <- resamples(
  list(glmn = glmn_mod, MARS = mars_mod,  bagged = mars_gcv_bag)
)

# Slide 57

library(tidyposterior)
rmse_mod <- perf_mod(rs, seed = 4344, iter = 5000, metric = "RMSE")

# Slide 58

posteriors <- tidy(rmse_mod, seed = 366784)
summary(posteriors)

ggplot(posteriors)

# Slide 59

differences <-
  contrast_models(
    rmse_mod,
    list_1 = rep("bagged", 2),
    list_2 = c("glmn", "MARS"),
    seed = 2581
  )

ggplot(differences, size = 0.25)

# Slide 60

summary(differences, size = 0.25)

# Slide 62

car_test <- car_test %>%
  mutate(pred = predict(mars_gcv_bag, car_test))

library(yardstick)
rmse(car_test, truth = mpg, estimate = pred)

ggplot(car_test, aes(x = mpg, y = pred)) +
  geom_abline(col = "green", alpha = .5) + 
  geom_smooth(se = FALSE, col = "red", 
              lty = 2, lwd = 1, alpha = .5) + 
  geom_point(alpha = .3)

# Stop the cluster if you used it

stopCluster(cl)

