# Packages 
library(ggplot2)
library(dplyr)

# Load data
df <- read.csv('training_data/DQ_pink4.csv')

# Plots
ggplot(data=df) +
    geom_point(aes(x=X, y=loss)) +
    labs(title='Exploding Loss Over 1750 Episodes',
    x='Episode Number', y='Loss')
ggsave('loss_plot.png', width=7, height=5)

# Statistics
df %>%
    filter(loss == min(loss))

colnames(df)
nrow(df)
