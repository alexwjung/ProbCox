# -----------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
library(survival)
setwd('/Users/alexwjung/Documents/ProbCox/paper/ProbCox/')
data("heart")
heart = heart[complete.cases(heart),]

#model.matrix( ~ differ - 1, data=colon)

heart = data.frame(c(heart['start'], heart['stop'], heart['event'], heart['age'], heart['year'], heart['surgery'],  heart['transplant']))

standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}
names(heart)
heart[ , c(4, 5)] = apply(heart[ , c(4, 5)], 2, standardize)

write.csv(heart, './data/application/heart.csv')

m = coxph(Surv(start, stop, event) ~., data=heart)
summary(m)

x = paste(unname(m$coefficients), collapse="; ")
write(x, file='./out/application/heart/R_theta.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file='./out/application/heart/R_se.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file='./out/application/heart//R_concordance.txt', ncolumns = 1, append = FALSE, sep = ";")
