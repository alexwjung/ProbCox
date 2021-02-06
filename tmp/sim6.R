rm(list=ls())
library(survival)
ROOT_DIR = getwd()

sim_name = 'sim6'

sim <- read.csv(paste(ROOT_DIR, '/tmp/', sim_name, '.csv', sep='') , header=FALSE, sep=";")

sim <- as.data.frame(as.matrix(sim))
m = coxph(Surv(V1, V2, V3) ~., data=sim)

x = paste(unname(m$concordance['concordance']), collapse=" ")
write(x, file = paste(ROOT_DIR, '/output/simulation/', sim_name, '/R_concordance.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

x = paste(unname(m$coefficients), collapse="; ")
write(x, file = paste(ROOT_DIR, '/output/simulation/', sim_name, '/R_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

x = paste(sqrt(diag(m$var)), collapse="; ")
write(x, file = paste(ROOT_DIR, '/output/simulation/', sim_name, '/R_se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
