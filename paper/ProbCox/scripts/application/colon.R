# -----------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
library(survival)
setwd('/Users/alexwjung/Documents/ProbCox/paper/ProbCox/')
data("colon")
colon = colon[complete.cases(colon),]

colon = data.frame(c(colon['time'], colon['status'], colon['sex'], colon['age'], colon['obstruct'], colon['perfor'], colon['adhere'], colon['nodes'], colon['node4'], colon['surg'], colon['extent'], colon['differ']))

standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}

colon[1:5,]
#names(colon)
colon[ , c(4, 8, 11, 12)] = apply(colon[ , c(4, 8, 11, 12)], 2, standardize)

write.csv(colon, './data/application/colon.csv')


m = coxph(Surv(time, status) ~., data=colon)
summary(m)

x = paste(unname(m$coefficients), collapse="; ")
write(x, file='./out/application/colon/R_theta.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file='./out/application/colon/R_se.txt', ncolumns = 1, append = FALSE, sep = ";")

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file='./out/application/colon//R_concordance.txt', ncolumns = 1, append = FALSE, sep = ";")
