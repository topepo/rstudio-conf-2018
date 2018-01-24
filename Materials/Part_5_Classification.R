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

library(yardstick)
library(dplyr)
two_class_example %>% head(4)

# Slide 5

two_class_example %>% 
	conf_mat(truth = truth, estimate = predicted)

two_class_example %>% 
	accuracy(truth = truth, estimate = predicted)

# Slide 6

two_class_example %>% 
  sens(truth = truth, estimate = predicted)

two_class_example %>% 
  spec(truth = truth, estimate = predicted)

# Slide 9

library(pROC)
roc_obj <- roc(
	response = two_class_example$truth,
	predictor = two_class_example$Class1,
	# If the first level is the event of interest:
	levels = rev(levels(two_class_example$truth))
	)

auc(roc_obj)

plot(
	roc_obj,
	legacy.axes = TRUE,
	print.thres = c(.2, .5, .8), 
	print.thres.pattern = "cut = %.2f (Spec = %.2f, Sens = %.2f)",
	print.thres.cex = .8
)

# Slide 11

library(dplyr)
load("Data/okc.RData")
okc_train %>% dim()
okc_test %>% nrow()
table(okc_train$Class)

# Slide 16

library(caret)
ctrl <- trainControl(
	method = "cv",
	# Also predict the probabilities
	classProbs = TRUE,
	# Compute the ROC AUC as well as the sens and  
	# spec from the default 50% cutoff. The 
	# function `twoClassSummary` will produce those. 
	summaryFunction = twoClassSummary,
	savePredictions = "final",
	sampling = "down"
)

# Slide 22

set.seed(5515)
cart_mod <- train(
	x = okc_train[, names(okc_train) != "Class"], 
	y = okc_train$Class,
	method = "rpart2",
	metric = "ROC",
	tuneGrid = data.frame(maxdepth = 1:20),
	trControl = ctrl
)

# Slide 23

cart_mod$finalModel

# Slide 24

ggplot(cart_mod)

# Slide 25

plot_roc <- function(x, ...) {
	averaged <- x %>%
		group_by(rowIndex, obs) %>%
		summarise(stem = mean(stem, na.rm = TRUE))
	roc_obj <- roc(
		averaged[["obs"]], 
		averaged[["stem"]], 
		levels = rev(levels(averaged$obs))
	)
	plot(roc_obj, ...)
}
plot_roc(cart_mod$pred)

# Slide 26

confusionMatrix(cart_mod)

# Slide 27

cart_imp <- varImp(cart_mod, scale = FALSE, 
                   surrogates = FALSE, competes = FALSE)
ggplot(cart_imp, top = 7) + xlab("")

# Slide 35

set.seed(5515)
cart_bag <- train(
	x = okc_train[, names(okc_train) != "Class"], 
	y = okc_train$Class,
	method = "treebag",
	metric = "ROC",
	trControl = ctrl
)

# Slide 36

cart_bag

# Slide 37

confusionMatrix(cart_bag)

# Slide 38

plot_roc(cart_mod$pred)
plot_roc(cart_bag$pred, col = "darkred", add = TRUE)

# Slide 39

bag_imp <- varImp(cart_bag, scale = FALSE)
ggplot(bag_imp, top = 30) + xlab("")


# Slide 46

ggplot(okc_train, aes(x = essay_length, col = Class)) + 
	geom_line(stat = "density")

# Slide 48

library(tidyr)

pred_xtab <- table(okc_train$religion, okc_train$Class)
pred_xtab <- t(apply(pred_xtab, 2, function(x) x/sum(x)))
pred_xtab <- as.data.frame(pred_xtab) %>%
	tibble::rownames_to_column("Class") %>%
	gather(value, prob, -Class)

ggplot(pred_xtab, aes(x = reorder(value, prob), y = prob, fill = Class)) + 
	geom_bar(stat = "identity", position = position_dodge()) + 
	xlab("") + 
	ylab("Within-Class Probability")

# Slide 53

library(recipes)
is_dummy <- vapply(okc_train, function(x) length(unique(x)) == 2 & is.numeric(x), logical(1))
dummies <- names(is_dummy)[is_dummy]
no_dummies <- recipe(Class ~ ., data = okc_train) %>%
	step_bin2factor(!!! dummies) %>%
	step_zv(all_predictors())

smoothing_grid <- expand.grid(usekernel = TRUE, fL = 0, adjust = seq(0.5, 3.5, by = 0.5))

# Slide 54

set.seed(5515)
nb_mod <- train(
	no_dummies,
	data = okc_train,
	method = "nb",
	metric = "ROC",
	tuneGrid = smoothing_grid,
	trControl = ctrl
)

# Slide 55

ggplot(nb_mod)

# Slide 56

plot_roc(cart_mod$pred)
plot_roc(cart_bag$pred, col = "red", add = TRUE)
plot_roc(nb_mod$pred, col = "blue", add = TRUE)

# Slide 58

rs <- resamples(
  list(CART = cart_mod, Bagged = cart_bag, Bayes = nb_mod)
)
library(tidyposterior)
roc_mod <- perf_mod(rs, seed = 2560, iter = 5000)

# Slide 59

roc_dist <- tidy(roc_mod)
summary(roc_dist)
differences <-
	contrast_models(
		roc_mod,
		list_1 = c("Bagged", "Bayes"),
		list_2 = c("CART", "Bagged"),
		seed = 650
	)

# Slide 60

summary(differences, size = 0.025)

differences %>%
	mutate(contrast = paste(model_2, "vs", model_1)) %>%
	ggplot(aes(x = difference, col = contrast)) + 
	geom_line(stat = "density") + 
	geom_vline(xintercept = c(-0.025, 0.025), lty = 2)

# Slide 62

test_res <- okc_test %>%
	dplyr::select(Class) %>%
	mutate(
		prob = predict(nb_mod, okc_test, type = "prob")[, "stem"],
		pred = predict(nb_mod, okc_test)
	)
roc_curve <- roc(test_res$Class, test_res$prob, levels = c("other", "stem"))


# Slide 63

plot(
	roc_curve,
	print.thres = .5,
	print.thres.pattern = "cut = %.2f (Sp = %.3f, Sn = %.3f)",
	legacy.axes = TRUE
)

roc_curve
getTrainPerf(nb_mod)

# Slide 64

ggplot(test_res, aes(x = prob)) + geom_histogram(binwidth = .04) + facet_wrap(~Class)

