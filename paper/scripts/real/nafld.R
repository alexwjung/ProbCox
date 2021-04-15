# -----------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
library(survival)
setwd('/Users/alexwjung/Google Drive/projects/ProbCox/')
data("nafld")
nafld1 = nafld1[complete.cases(nafld1),]

nafld1 = data.frame(c(nafld1['futime'], nafld1['status'], nafld1['bmi'], nafld1['height'], nafld1['weight'], nafld1['male'], nafld1['age']))

standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}
names(nafld1)

nafld1[ , c(3, 4, 5, 7)] = apply(nafld1[ , c(3, 4, 5, 7)], 2, standardize)

write.csv(nafld1, './data/real/nafld1.csv')

m = coxph(Surv(futime, status) ~., data=nafld1)
summary(m)

x = paste(unname(m$coefficients), collapse="; ")
write(x, file='./output/real/nafld1/R_theta.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file='./output/real/nafld1/R_se.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file='./output/real/nafld1//R_concordance.txt', ncolumns = 1, append = FALSE, sep = ";")
