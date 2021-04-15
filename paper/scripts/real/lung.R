# -----------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
library(survival)
setwd('/Users/alexwjung/Google Drive/projects/ProbCox/')
data("lung")
lung = lung[complete.cases(lung),]
lung$status = lung$status - 1
#lung$ph.ecog = as.factor(lung$ph.ecog)
lung$sex = lung$sex - 1
#lung = data.frame(c(lung['time'], lung['status'], lung['age'], lung['sex'], data.frame(model.matrix( ~ ph.ecog - 1, data=lung))['ph.ecog1'], data.frame(model.matrix( ~ ph.ecog - 1, data=lung))['ph.ecog2'], data.frame(model.matrix( ~ ph.ecog - 1, data=lung))['ph.ecog3'], lung['ph.karno'], lung['pat.karno'], lung['meal.cal'],lung['wt.loss']))
lung = data.frame(c(lung['time'], lung['status'], lung['age'], lung['sex'], lung['ph.ecog'], lung['ph.karno'], lung['pat.karno'], lung['meal.cal'],lung['wt.loss']))

standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}

lung[ , c(3, 6, 7, 8, 9)] = apply(lung[ , c(3, 6, 7, 8, 9)], 2, standardize)
#lung[ , c(3, 8, 9, 10, 11)] = apply(lung[ , c(3, 8, 9, 10, 11)], 2, standardize)

write.csv(lung, './data/real/lung.csv')

m = coxph(Surv(time, status) ~., data=lung)
summary(m)

x = paste(unname(m$coefficients), collapse="; ")
write(x, file='./output/real/lung/R_theta.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file='./output/real/lung/R_se.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file='./output/real/lung//R_concordance.txt', ncolumns = 1, append = FALSE, sep = ";")
