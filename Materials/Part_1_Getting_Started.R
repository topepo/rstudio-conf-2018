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

# Slide 12

library(tidyverse)

ames <- 
  read_delim("http://bit.ly/2whgsQM", delim = "\t") %>%
  rename_at(vars(contains(' ')), funs(gsub(' ', '_', .))) %>%
  rename(Sale_Price = SalePrice) %>%
  filter(!is.na(Electrical)) %>%
  select(-Order, -PID, -Garage_Yr_Blt)

ames %>%
  group_by(Alley) %>%
  summarize(mean_price = mean(Sale_Price/1000),
            n = sum(!is.na(Sale_Price)))

# Slide 13

ggplot(ames, 
       aes(x = Garage_Type,
           y = Sale_Price)) + 
  geom_violin() + 
  coord_trans(y = "log10") + 
  xlab("Garage Type") + 
  ylab("Sale Price") 



# Slide 14

library(purrr)

# summarize via purrr::map
by_alley <- split(ames, ames$Alley)
is_list(by_alley)

map(by_alley, nrow)

# or better yet:
map_int(by_alley, nrow)


# works on non-list vectors too
ames %>%
	mutate(Sale_Price = Sale_Price %>%
				 	map_dbl(function(x) x / 1000)) %>%
	select(Sale_Price, Yr_Sold) %>%
	head(4)

# Slide 15

library(AmesHousing)
ames <- make_ames()

