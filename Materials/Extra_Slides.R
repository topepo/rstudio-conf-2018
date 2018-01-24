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


# Create Data for PCA slides
library(caret)
data(segmentationData)

segmentationData <- segmentationData[, c("EqSphereAreaCh1", "PerimCh1", "Class", "Case")]
names(segmentationData)[1:2] <- paste0("Predictor", LETTERS[1:2])

segmentationData$Class <- factor(ifelse(segmentationData$Class == "PS", "One", "Two"))

bivariate_data_train <- subset(segmentationData, Case == "Train")
bivariate_data_test  <- subset(segmentationData, Case == "Test")

bivariate_data_train$Case <- NULL
bivariate_data_test$Case  <- NULL

# Slide 4

library(recipes)

bivariate_rec <- recipe(Class ~ PredictorA + PredictorB, 
                         data = bivariate_data_train) %>%
  step_BoxCox(all_predictors())

bivariate_rec <- prep(bivariate_rec, training = bivariate_data_train, verbose = FALSE)

inverse_test <- bake(bivariate_rec, newdata = bivariate_data_test, everything())

# Slide 8

ggplot(inverse_test, 
       aes(x = 1/PredictorA, 
           y = 1/PredictorB,
           color = Class)) +
  geom_point(alpha = .3, cex = 1.5) + 
  theme(legend.position = "top") +
  xlab("1/A") + ylab("1/B") 

# Slide 9 

bivariate_pca <- 
  recipe(Class ~ PredictorA + PredictorB, data = bivariate_data_train) %>%
  step_BoxCox(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(all_predictors()) %>%
  prep(training = bivariate_data_test, verbose = FALSE)

pca_test <- bake(bivariate_pca, newdata = bivariate_data_test)

# Put components axes on the same range
pca_rng <- extendrange(c(pca_test$PC1, pca_test$PC2))

ggplot(pca_test, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = .2, cex = 1.5) + 
  theme(legend.position = "top") +
  xlim(pca_rng) + ylim(pca_rng) + 
  xlab("Principal Component 1") + ylab("Principal Component 2") 


# Data for Remaining Slides

library(AmesHousing)
ames <- make_ames()
nrow(ames)

library(rsample)

# Make sure that you get the same random numbers
set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price")

ames_train <- training(data_split)
ames_test  <- testing(data_split)

set.seed(2453)
cv_splits <- vfold_cv(ames_train, v = 10, strata = "Sale_Price")

lin_coords <- recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
                      Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
                      Central_Air + Longitude + Latitude,
                    data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = 0.05)  %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) 

coords <- lin_coords %>%
  step_bs(Longitude, Latitude, options = list(df = 5))

library(purrr)
cv_splits <- cv_splits %>% 
  mutate(coords = map(splits, prepper, recipe = coords, retain = TRUE))

lm_fit_rec <- function(rec_obj, ...) 
  lm(..., data = juice(rec_obj))

cv_splits <- cv_splits %>% 
  mutate(fits = map(coords, lm_fit_rec, Sale_Price ~ .))

# Slide 12

coef_summary <- map(cv_splits$fits, tidy) %>% bind_rows
head(coef_summary)

# Slide 13

num_pred <- c("Year_Built", "Gr_Liv_Area", 
              "Full_Bath", "Year_Sold", 
              "Lot_Area")
z_lim <- qnorm(c(0.025, 0.975))
coef_summary %>% 
  filter(term %in% num_pred) %>% 
  ggplot(aes(x = term, y = statistic)) + 
  geom_hline(yintercept = z_lim,
             lty = 2, col = "red") + 
  geom_jitter(width = .1, alpha = .4, cex = 2) + 
  coord_flip() + 
  xlab("") 

# Slide 14

coef_summary %>% 
  filter(grepl("^Neighborhood", term)) %>% 
  ggplot(aes(x = term, y = statistic)) + 
  geom_hline(
    yintercept = z_lim,
    lty = 2, 
    col = "red"
  ) + 
  geom_jitter(
    width = .1, 
    alpha = .4, 
    cex = 2
  ) + 
  coord_flip() + 
  xlab("") + 
  ylab(
    paste("Difference from", 
          levels(ames_test$Neighborhood)[1])
  ) 

# Slide 17

library(yardstick)
data(solubility_test)
str(solubility_test, nchar.max = 70)

# Slide 18

rmse(solubility_test, truth = solubility, estimate = prediction)

# Slide 19

rsq(solubility_test, truth = solubility, estimate = prediction)

# Slide 20

library(dplyr)
set.seed(3545)
solubility_test <- solubility_test %>% mutate(noise = sample(prediction))

rsq(solubility_test, truth = solubility, estimate = noise)
rsq_trad(solubility_test, truth = solubility, estimate = noise)

