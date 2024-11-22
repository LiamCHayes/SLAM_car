# Packages 
library(ggplot2)
library(dplyr)

# Load data
df <- read.csv('training_data/DQ_pink3.csv')

# Plots
ggplot(data=df) +
    geom_point(aes(x=X, y=pct_explored)) 

# Statistics
df %>%
    filter(loss == min(loss))

colnames(df)
nrow(df)
