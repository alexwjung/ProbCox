# -----------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
library(survival)
setwd('/Users/alexwjung/Google Drive/projects/ProbCox/')
data("mgus")
mgus1 = mgus1[complete.cases(mgus1),]

mgus1 = data.frame(c(mgus1['start'], mgus1['stop'], mgus1['status'], mgus1['mspike'], mgus1['hgb'], mgus1['creat'], mgus1['alb'], mgus1['dxyr'], data.frame(model.matrix( ~ sex - 1, data=mgus1))['sexfemale'], data.frame(model.matrix( ~ pcdx - 1, data=mgus1))['pcdxAM'], data.frame(model.matrix( ~ pcdx - 1, data=mgus1))['pcdxLP'], data.frame(model.matrix( ~ pcdx - 1, data=mgus1))['pcdxMA']))


standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}
names(mgus1)

mgus1[ , c(4, 5, 6, 7, 8)] = apply(mgus1[ , c(4, 5, 6, 7, 8)], 2, standardize)

write.csv(mgus1, './data/real/mgus1.csv')


m = coxph(Surv(start, stop, status) ~., data=mgus1)
summary(m)

x = paste(unname(m$coefficients), collapse="; ")
write(x, file='./output/real/mgus1/R_theta.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file='./output/real/mgus1/R_se.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file='./output/real/mgus1//R_concordance.txt', ncolumns = 1, append = FALSE, sep = ";")
