# PROGETTO ESAME STREAMING DATA MANAGEMENT AND TIME SERIES 

rm(list=ls())
# LIBRERIE -----

library(readr)
library(dplyr)
library(xts)
library(imputeTS)
library(forecast)
library(MLmetrics)
library(ggplot2)
library(ggpubr)
library(KFAS)
library(tsfknn)

Sys.setenv(TZ = "UTC")

# DATI ------

# Viene fornita una time series univariata, relativa a misurazioni orarie di ossido di carbonio (CO). 
# I dati sono organizzati nelle seguenti 3 colonne:
#   
# Date - stringa codificante la data della misurazione, in formato yyyy-mm-dd
# Hour - intero indicante l'ora della misurazione. I valori vanno da 0 a 23, 
#        con 0 che rappresenta l'intervallo 00:00 - 00:59, 1 che rappresenta 
#        l'intervallo 01:00 - 01:59, ... e 23 che rappresenta l'intervallo 23:00 - 23:59
# CO   - valore di CO rilevato

# I dati coprono il periodo da 2004-03-10 (hour=18) a 2005-02-28 (hour=23)

df <- read_csv("C:\\Users\\emanu\\Downloads\\Project_data_2021_2022 (TRAINSET).csv", 
                    col_types = cols(Date = col_character()))


View(df)

# ANALISI ESPLORATIVA -----

summary(df)

str(df)

# verifica duplicati

anyDuplicated(df)

# nessun duplicato

# verifica di consistenza 

df %>% 
  group_by(Date) %>%
  summarise(freq=n()) %>% 
  group_by(freq) %>% 
  count()

# va bene

#giorni della settimana

df$weekday <- weekdays(as.Date(df$Date))
df$weekday <- factor(df$weekday, levels=c("Monday", "Tuesday", 
                                                       "Wednesday","Thursday",
                                                       "Friday","Saturday","Sunday"))
ggplot(df) +
  aes(x = weekday, y = CO) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()
  
# types

df$datetime <- paste(df$Date, df$Hour, sep=' ')
df$datetime <- as.POSIXct(df$datetime, format='%Y-%m-%d %H', tz="UTC")

#tz serve per disattivare il cambio d'ora (già sistemato)

class(df$datetime)

#conversione a time series

ts_base <- xts(df$CO, df$datetime)
names(ts_base) <- 'C0'

plot(ts_base, main='Valori di CO', type='l', cex=2)

# outliers

out_ts<- tsoutliers(ts_base)
length(out_ts$index)

#no outliers

# NA values

apply(is.na(df), 2, sum)
idx_na <- apply(is.na(df), 2, which)

df[idx_na$CO,]

365/nrow(df)*100 # 4.28 % circa di missing values sul totale

ggplot_na_distribution(ts_base)

ggplot_na_gapsize(ts_base)

ggplot_na_gapsize(ts_base, ranked_by = 'total')

ggplot_na_intervals(ts_base,interval_size = 720) #720 è dato da un mese (24*30)

#Stagionalità

# giornaliera
ts_daily <- ts_base %>%
  ts(frequency = 24)  

ggseasonplot(ts_daily[, "CO"],                      #-- Stagionalità Giornaliera
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "CO",              #-- Etichetta Asse Y
             xlab = "Orario del Giorno",                #-- Etichetta Asse X
             main = "Grafico Stagionale Giornaliero") + #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45)) +       #-- 45?
  scale_y_continuous(breaks = c(500,
                                1000,
                                1500,
                                2000),
                     ) +             #-- Etichetta
  theme_bw() 

#settimanale

ts_weekly <- ts_base %>%
  ts(frequency = 24*7)  

ggseasonplot(ts_weekly[, "C0"],                      #-- Stagionalità Giornaliera
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "C0",              #-- Etichetta Asse Y
             xlab = "Ora settimanale",                #-- Etichetta Asse X
             main = "Grafico Stagionale settimale") + #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45)) +       #-- 45
  scale_y_continuous(breaks = c(500,
                                1000,
                                1500,
                                2000)) +#-- Etichetta
  theme_bw()

#DECOMPOSE

multis_ts_base <- msts(ts_base, seasonal.periods=c(24, 24*7))
multis_ts_base %>% 
  mstl() %>%
  autoplot() + theme_bw()

# ACF e PACF

ggarrange(ggAcf(ts_base$C0,
                lag.max = 168,        #-- 2 Days for Seasonality
                main = ""),
          ggPacf(ts_base$C0,
                 lag.max = 168,       #-- 2 Days for Seasonality
                 main = ""),
          labels = c("ACF", "PACF"),
          nrow = 2, ncol= 1)

ts_base <- ts(ts_base, frequency = 24) #abbiamo stagionalità giornaliera

# IMPUTAZIONE -----

ts_base_train <- ts_base[df$datetime<'2005-02-22 00:00:00']
ts_base_val <- ts_base[df$datetime>='2005-02-22 00:00:00']

# è importante notare come il validation set sia stato scelto in modo tale che esso
# non abbia valori nulli

ggplot_na_distribution(ts_base_train)
ggplot_na_distribution(ts_base_val)

# metodi di imputazione ---
# interpolazione lineare

ts_imp_linear <- na_interpolation(ts_base_train, option='linear')

ggplot_na_imputations(ts_base_train, ts_imp_linear)
ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_linear[7950:8150])

tbats_linear <- tbats(ts_imp_linear, use.parallel = T)
forecast_linear <- forecast(tbats_linear, h=336)
plot(forecast_linear, ylim=c(0,2000))

MAPE(forecast_linear$mean, ts_base_val)
rmse(forecast_linear$mean, ts_base_val)



# kalman
ts_imp_kalman <- na_kalman(ts_base_train, model = 'auto.arima', smooth = T)
ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_kalman[7950:8150])

tbats_kalman <- tbats(ts_imp_kalman, use.parallel = T)
forecast_kalman <- forecast(tbats_kalman, h=336)
plot(forecast_kalman,ylim=c(0,2000))

MAPE(forecast_kalman$mean, ts_base_val)
rmse(forecast_kalman$mean, ts_base_val)

# ma
ts_imp_ma<- na_ma(ts_base_train, weighting = 'exponential')
ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_ma[7950:8150])

tbats_ma <- tbats(ts_imp_ma, use.parallel = T)
forecast_ma <- forecast(tbats_ma, h=336)
plot(forecast_ma,ylim=c(0,2000))
MAPE(forecast_ma$mean, ts_base_val)
rmse(forecast_ma$mean, ts_base_val)

#seadec interpolation
ts_imp_seadec_int<- na_seadec(ts_base_train, algorithm = 'interpolation', find_frequency = T)
ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seadec_int[7950:8150])

tbats_seadec_int <- tbats(ts_imp_seadec_int, use.parallel = T)
forecast_seadec_int <- forecast(tbats_seadec_int, h=336)
plot(forecast_seadec_int,ylim=c(0,2000))

MAPE(forecast_seadec_int$mean, ts_base_val)
rmse(forecast_seadec_int$mean, ts_base_val)

ts_imputations <- na_seadec(ts_base, algorithm = 'interpolation', find_frequency = T)
df_to_export <- cbind(df[,c(1,2,4)], as.vector(ts_imputations))
colnames(df_to_export)[4] <- 'CO'


write_csv(df_to_export, file = "C:\\Users\\emanu\\Downloads\\imputed_ts.csv")


# #seadec kalman
# ts_imp_seadec_kalman<- na_seadec(ts_base_train, algorithm = 'kalman', find_frequency = T)
# ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seadec_kalman[7950:8150])
# 
# tbats_seadec_kalman <- tbats(ts_imp_seadec_kalman, use.parallel = T)
# forecast_seadec_kalman <- forecast(tbats_seadec_kalman, h=336)
# plot(forecast_seadec_kalman,ylim=c(0,2000))
# 
# MAPE(forecast_seadec_kalman$mean, ts_base_val)
# 
# #seadec ma
# ts_imp_seadec_ma<- na_seadec(ts_base_train, algorithm = 'ma', find_frequency = T)
# ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seadec_ma[7950:8150])
# 
# tbats_seadec_ma <- tbats(ts_imp_seadec_ma, use.parallel = T)
# forecast_seadec_ma <- forecast(tbats_seadec_ma, h=336)
# plot(forecast_seadec_ma,ylim=c(0,2000))
# 
# MAPE(forecast_seadec_ma$mean, ts_base_val)
# 
# #seasplit int
# ts_imp_seasplit_int<- na_seasplit(ts_base_train, algorithm = 'interpolation', find_frequency = T)
# ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seasplit_int[7950:8150])
# 
# tbats_seasplit_int <- tbats(ts_imp_seasplit_int, use.parallel = T)
# forecast_seasplit_int <- forecast(tbats_seasplit_int, h=336)
# plot(forecast_seasplit_int,ylim=c(0,2000))
# 
# MAPE(forecast_seasplit_int$mean, ts_base_val)
# 
# #seasplit kalman
# ts_imp_seasplit_kalman<- na_seasplit(ts_base_train, algorithm = 'kalman', find_frequency = T)
# ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seasplit_kalman[7950:8150])
# 
# tbats_seasplit_kalman <- tbats(ts_imp_seasplit_kalman, use.parallel = T)
# forecast_seasplit_kalman <- forecast(tbats_seasplit_kalman, h=336)
# plot(forecast_seasplit_kalman,ylim=c(0,2000))
# 
# MAPE(forecast_seasplit_kalman$mean, ts_base_val)
# 
# #seasplit ma
# ts_imp_seasplit_ma<- na_seasplit(ts_base_train, algorithm = 'ma', find_frequency = T)
# ggplot_na_imputations(ts_base_train[7950:8150], ts_imp_seasplit_ma[7950:8150])
# 
# tbats_seasplit_ma <- tbats(ts_imp_seasplit_ma, use.parallel = T)
# forecast_seasplit_ma <- forecast(tbats_seasplit_ma, h=336)
# plot(forecast_seasplit_ma,ylim=c(0,2000))
# 
# MAPE(forecast_seasplit_ma$mean, ts_base_val)

# ARIMA METHODS -------



#write_csv(df_to_export, file = "/Users/guglielmo/Desktop/SDMTS_PROJECT/data/imputed_ts.csv")

# riguardare miglior metodo di imputazione

train_arima <- head(ts_imputations, round(length(ts_imputations) * 0.8))
h <- length(ts_imputations) - length(train_arima)
val_arima <- tail(ts_imputations, h)

# train (start: 2004-03-10 18:00 ---> end: 2004-12-19 22:00)
# validation (start: 2004-12-19 23:00 ---> end: 2005-02-28 23:00)

#acf e pacf
ggarrange(ggAcf(ts_base,
                lag.max = 168, #-- 2 Days for Seasonality
                main = "ACF"),
          ggPacf(ts_base,
                 lag.max = 168, #-- 2 Days for Seasonality
                 main = "PACF"),
          nrow = 2, ncol= 1) + theme_gray()

# box-cox transformation 
BoxCox.lambda(train_arima) #approssimare lambda a -1

lambda_arima <- BoxCox.lambda(train_arima) 

train_arima_transformed <- BoxCox(train_arima, lambda_arima)
val_arima_transformed <- BoxCox(val_arima, lambda_arima)

#tseasonal difference
train_diff_seasonal<-diff(train_arima_transformed, lag=24)
autoplot(train_diff_seasonal, main='seasonal difference')+theme_gray()

Box.test(train_diff_seasonal, type='Ljung-Box') #H0 rejected

#first differences
train_diff_trend<-diff(train_diff_seasonal, differences = 1)
autoplot(train_diff_trend, main='first differences')+theme_gray() #c'è ancora stagionalità giornaliera
ggarrange(ggAcf(train_diff_trend, 
                lag.max = 72, main = "ACF"),
          ggPacf(train_diff_trend, 
                 lag.max = 72, main = "PACF"),
          nrow = 2, ncol= 1)+theme_gray()

Box.test(train_diff_trend, type='Ljung-Box') #H0 accepted

#1 differenza stagionale e 1 differenza semplice


#arima 1

mod1_arima <- Arima(train_arima, c(4,1,1), c(0,1,2), lambda = 'auto',
                    include.constant = TRUE)
checkresiduals(mod1_arima, plot=F)
summary(mod1_arima)

mod1_arima$aic
ggarrange(ggAcf(mod1_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(mod1_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

mod1_arima_forecast <- forecast(mod1_arima, h = 1705)
MAPE(mod1_arima_forecast$mean, val_arima)

plot(ts(val_arima))+lines(ts(mod1_arima_forecast$mean), type='l',col='blue')

#auto-arima

auto_arima <- auto.arima(train_arima, d=1, D=1, seasonal = T, lambda = 'auto')
summary(auto_arima)

ggarrange(ggAcf(auto_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(auto_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

checkresiduals(auto_arima, plot=F)

auto_arima_forecast <- forecast(auto_arima, h=length(val_arima))
MAPE(auto_arima_forecast$mean, val_arima)

plot(ts(val_arima))+lines(ts(mod1_arima_forecast$mean), type='l',col='red')


#### sono arrivato fin qui


#arima with fourier

train_arima_fourier <- msts(train_arima, c(24, 7*24)) #-- Multistagionalità

mod2_arima <- Arima(train_arima_fourier, c(5, 1, 1), lambda = lambda_arima,
                    xreg = fourier(train_arima_fourier, K = c(3,3)))
summary(mod2_arima)


#include.constant = TRUE
ggarrange(ggAcf(mod2_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(mod2_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

checkresiduals(mod2_arima)

mod2_arima_forecast <- forecast(mod2_arima, xreg=fourier(train_arima_fourier, K=c(3,3), h=1705))

MAPE(ts(mod2_arima_forecast$mean, frequency = 24), as.vector(val_arima))

plot(ts(val_arima))+lines(ts(mod2_arima_forecast$mean), type='l',col='blue')

#mod4
mod4_arima <- Arima(train_arima, c(5,1,1), c(0,1,1), lambda = lambda_arima,
                    include.constant = TRUE)
checkresiduals(mod4_arima, plot=F)
summary(mod4_arima)

mod4_arima$aic
ggarrange(ggAcf(mod4_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(mod4_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

mod4_arima_forecast <- forecast(mod4_arima, h = 1705)
MAPE(mod4_arima_forecast$mean, val_arima)

plot(ts(val_arima))+lines(ts(mod4_arima_forecast$mean), type='l',col='red')

#mod5
mod5_arima <- Arima(train_arima, c(5,1,1), c(0,1,2),
                    include.constant = TRUE)
checkresiduals(mod5_arima, plot=F)
summary(mod5_arima)

mod5_arima$aic
ggarrange(ggAcf(mod5_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(mod5_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

mod5_arima_forecast <- forecast(mod5_arima, h = 1705)
MAPE(mod5_arima_forecast$mean, val_arima)

plot(ts(val_arima))+lines(ts(mod4_arima_forecast$mean), type='l',col='red')

#mod5
mod6_arima <- Arima(train_arima, c(4,1,1), c(0,1,2),
                    include.constant = TRUE)
checkresiduals(mod6_arima, plot=F)
summary(mod6_arima)

mod6_arima$aic
ggarrange(ggAcf(mod6_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(mod6_arima$residuals,
                 lag.max = 200,       #-- 2 Days for Seasonality
                 main = ""),
          nrow = 2, ncol= 1)

mod6_arima_forecast <- forecast(mod6_arima, h = 1705)
MAPE(mod6_arima_forecast$mean, val_arima)

plot(ts(val_arima))+lines(ts(mod6_arima_forecast$mean), type='l',col='red')

#best arima

#sub_ts_imputations <- ts(ts_imputations[1:8454], frequency=24)
#plot(sub_ts_imputations)

best_arima <- Arima(ts_imputations, c(4,1,1), c(0,1,2), lambda='auto', include.constant = T)
summary(best_arima)
best_arima$aic
ggarrange(ggAcf(best_arima$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(best_arima$residuals,
                 lag.max = 200,       
                 main = ""),
          nrow = 2, ncol= 1)

best_arima_forecast <- forecast(best_arima, h = 744)

best_arima_forecast$mean
plot(best_arima_forecast)







#best_fourier
best_fourier_ts <- msts(ts_imputations, c(24,7*24))

best_fourier <- Arima(best_fourier_ts,
                      c(4,1,1),
                      xreg=fourier(best_fourier_ts, K=c(2,2)),
                      
                      lambda = 'auto')

checkresiduals(best_fourier, plot=F)
summary(best_arima)
best_arima$aic

ggarrange(ggAcf(best_fourier$residuals,
                lag.max = 200,       
                main = ""),
          ggPacf(best_fourier$residuals,
                 lag.max = 200,       
                 main = ""),
          nrow = 2, ncol= 1)

best_four_forecast <-  forecast(best_fourier, xreg= fourier(best_fourier_ts, K=c(2,2), 744), 744)



                 
plot(best_four_forecast$mean, ylim=c(500,2000),col='blue')


# UCM -----

train_ucm <- head(ts_imputations, round(length(ts_imputations) * 0.8))
h <- length(ts_imputations) - length(train_ucm)
val_ucm <- tail(ts_imputations, h)

lambda_ucm <- BoxCox.lambda(train_ucm)

var_train_ucm <- var(train_ucm) 

train_ucm <- BoxCox(train_arima, lambda_ucm)


#var_train_ucm <- var(train_ucm) 

# LLT

mod1_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(NA,NA)), 
                    H = NA) 
mod1_ucm$Q

fit1_ucm <- fitSSM(mod1_ucm,
                   inits = log(c(var_train_ucm/1,
                                 var_train_ucm/1,
                                 var_train_ucm/50)))
fit1_ucm$optim.out$convergence #-- convergence

#-- previsioni

pred1_ucm <- predict(fit1_ucm$model, n.ahead = length(val_ucm))
pred1_ucm

plot(ts(val_ucm), main = "LLT previsions on validation", type = "l")
lines(ts(InvBoxCox(pred1_ucm, lambda=lambda_ucm)), col = "red")

InvBoxCox(pred1_ucm, lambda=lambda_ucm)
MAPE(InvBoxCox(pred1_ucm, lambda=lambda_ucm), val_ucm)

# LLT con seasonal dummy

mod2_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(NA, NA)) +
                      SSMseasonal(24, NA, sea.type = 'dummy'),
                    H = NA)

mod2_ucm$Q

fit2_ucm<- fitSSM(mod2_ucm,
                  inits = log(c(var_train_ucm/0.1,
                                var_train_ucm/1,
                                var_train_ucm/10,
                                var_train_ucm/20,
                                var_train_ucm/50,
                                var_train_ucm/100,
                                var_train_ucm/1000))) 

fit2_ucm$optim.out$convergence #convergence

#-- prevision

pred2_ucm <- predict(fit2_ucm$model,
                     n.ahead = length(val_ucm))
plot(ts(val_ucm), main = "LLT with seasonal dummy", type = "l")
lines(ts(InvBoxCox(pred2_ucm, lambda=lambda_ucm)), col = "blue")

MAPE(InvBoxCox(pred2_ucm,lambda=lambda_ucm), val_ucm)

## LLT + seasonal dummy + cycle (weekly)

mod3_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(NA, NA)) +
                      SSMseasonal(24, NA, sea.type = 'dummy') +
                      SSMcycle(24*7),
                    H = NA)

fit3_ucm<- fitSSM(mod3_ucm,
                  inits = log(c(var_train_ucm/0.1,
                                var_train_ucm/1,
                                var_train_ucm/10,
                                var_train_ucm/20,
                                var_train_ucm/50,
                                var_train_ucm/100,
                                var_train_ucm/1000))) 
fit3_ucm$optim.out$convergence #-- raggiunto num max di operazioni

#-- previsioni
pred3_ucm <- predict(fit3_ucm$model,
                     n.ahead = length(val_ucm)) #-- Numerosità Validation (3504)

plot(ts(val_ucm), main = "LLT with dummy with cycle", type = "l")
lines(ts(InvBoxCox(pred3_ucm, lambda=lambda_ucm)), col = "blue")

MAPE(InvBoxCox(pred3_ucm,lambda=lambda_ucm), val_ucm)

# LLT + trigonometric

mod4_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(NA, NA)) +
                      SSMseasonal(24, NA, sea.type = "trigon"),
                    H = NA)

mod4_ucm$Q

updt4 <- function(pars, model){
  model$Q[1, 1, 1] <- InvBoxCox(pars[1],lambda=lambda_ucm)
  model$Q[2, 2, 1] <- InvBoxCox(pars[2],lambda=lambda_ucm)
  diag(model$Q[3 : 25, 3 : 25, 1]) <- InvBoxCox(pars[3],lambda=lambda_ucm)
  model$H[1, 1, 1] <- InvBoxCox(pars[4],lambda=lambda_ucm)
  model
}

fit4_ucm <- fitSSM(mod4_ucm,
                   BoxCox(c(var_train_ucm/1000,
                         var_train_ucm/1000,
                         var_train_ucm/1000,
                         var_train_ucm/100000), lambda=lambda_ucm),
                   updt4,
                   control = list(maxit = 1000))

fit4_ucm$optim.out$convergence 

#-- prevision
pred4_ucm <- predict(fit4_ucm$model,
                     n.ahead = length(val_ucm)) 

lambda_ucm
pred4_ucm
plot(ts(val_ucm), main = "Prevision LLT + s.trigo", type = "l", ylab='validation', xlab='time')
lines(ts(InvBoxCox(pred4_ucm, lambda=lambda_ucm)), col = "red")

InvBoxCox(pred4_ucm, lambda=lambda_ucm)

#-- mape
MAPE(InvBoxCox(pred4_ucm, lambda=lambda_ucm), val_ucm)


# sono arrivato fin qui

# RW
mod5_ucm <- SSModel(train_ucm ~ SSMtrend(1, NA) +
                      SSMseasonal(24, NA, sea.type = 'dummy'),
                    H = NA)

fit5_ucm<- fitSSM(mod5_ucm,
                  inits = log(c(var_train_ucm/0.1,
                                var_train_ucm/1,
                                var_train_ucm/10,
                                var_train_ucm/20,
                                var_train_ucm/50,
                                var_train_ucm/100,
                                var_train_ucm/1000))) 

fit5_ucm$optim.out$convergence

#-- previsions
pred5_ucm <- predict(fit5_ucm$model,
                     n.ahead = length(val_ucm)) 

plot(ts(val_ucm), main = "RW with dummy", type = "l", ylab='validation', xlab='time')
lines(ts(InvBoxCox(pred5_ucm, lambda=lambda_ucm)), col = "blue")

#-- mae
MAPE(InvBoxCox(pred5_ucm, lambda=lambda_ucm), val_ucm)

#IRW
mod6_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(0, NA)) +
                      SSMseasonal(24, NA, sea.type = 'dummy') +
                      SSMcycle(24*7),
                    H = NA)

fit6_ucm<- fitSSM(mod6_ucm,
                  inits = log(c(var_train_ucm/0.1,
                                var_train_ucm/1,
                                var_train_ucm/10,
                                var_train_ucm/20,
                                var_train_ucm/50,
                                var_train_ucm/100,
                                var_train_ucm/1000))) 

fit6_ucm$optim.out$convergence

#-- previsions
pred6_ucm <- predict(fit6_ucm$model,
                     n.ahead = length(val_ucm)) 

pred6_ucm

plot(ts(val_ucm), main = "IRW", type = "l", ylab='validation', xlab='time')
lines(ts(InvBoxCox(pred6_ucm, lambda=lambda_ucm)), col = "blue")

#-- mae
MAPE(InvBoxCox(pred6_ucm, lambda=lambda_ucm), val_ucm)

#RWD
mod7_ucm <- SSModel(train_ucm ~ SSMtrend(2, list(NA, 0)) +
                      SSMseasonal(24, NA, sea.type = 'dummy') +
                      SSMcycle(24*7),
                    H = NA)

fit7_ucm<- fitSSM(mod7_ucm,
                  inits = log(c(var_train_ucm/0.1,
                                var_train_ucm/1,
                                var_train_ucm/10,
                                var_train_ucm/20,
                                var_train_ucm/50,
                                var_train_ucm/100,
                                var_train_ucm/1000))) 

fit7_ucm$optim.out$convergence

#-- previsions
pred7_ucm <- predict(fit7_ucm$model,
                     n.ahead = length(val_ucm)) 

pred7_ucm

plot(ts(val_ucm), main = "RW with dummy with cycle", type = "l", ylab='validation', xlab='time')
lines(ts(InvBoxCox(pred7_ucm, lambda=lambda_ucm)), col = "blue")

#-- mape
MAPE(InvBoxCox(pred7_ucm, lambda=lambda_ucm), val_ucm)

#ucm
sub_ts_imputations <- ts(ts_imputations[0:8454], frequency=24)
plot(sub_ts_imputations)

BoxCox.lambda(ts_imputations)
lambda_ucm

best_ucm <- SSModel(BoxCox(ts_imputations, lambda = -1) ~ SSMtrend(2, list(NA, NA)) +
                      SSMseasonal(24, NA, sea.type = 'dummy') +
                      SSMcycle(24*7),
                    H = NA)

best_fit_ucm<- fitSSM(best_ucm,
                      inits = log(c(var_train_ucm/1,
                                       var_train_ucm/1,
                                       var_train_ucm/1,
                                       var_train_ucm/10,
                                       var_train_ucm/20))) 
best_fit_ucm$optim.out$convergence #-- raggiunto num max di operazioni

#-- previsioni
best_pred_ucm <- predict(best_fit_ucm$model, n.ahead = 744) #-- Numerosità Validation (3504)

best_pred_ucm_inv <- InvBoxCox(best_pred_ucm,lambda=-1)
best_pred_ucm_inv
plot(val_ucm)
plot(merged_ts, col='blue')
length(best_pred_ucm)


merged_ts <- ts(c(val_ucm, best_pred_ucm_inv),               
                start = start(val_ucm),
                frequency = frequency(val_ucm))
plot(merged_ts, col='blue')


#grafico per ucm


merged_ts <- ts(c(ts_imputations, best_pred_ucm_inv),               
                start = start(ts_imputations),
                frequency = frequency(ts_imputations))

train_ucm
val_ucm

plot(merged_ts, col='blue')




















df_ml_new <- read_csv("C:\\Users\\emanu\\Downloads\\Telegram Desktop\\prediction_13_new_submission.csv", 
                      col_types = cols(Date = col_character()))


df_matricola_vecchia <- read_csv("C:\\Users\\emanu\\Downloads\\Telegram Desktop\\812503_20220108.csv", 
                                 col_types = cols(Date = col_character()))




df_matricola_nuova <- read_csv("C:\\Users\\emanu\\Downloads\\812503_20220117.csv", 
                                 col_types = cols(Date = col_character()))






#Si procede infine a creare il file con le previsioni da consegnare.


best_arima <- best_four_forecast$mean
#best_ucm <- best_pred_ucm_inv

best_ucm <- df_matricola_vecchia$UCM

best_ml <- round(as.numeric(df_ml_new$value))

test <-  data.frame(Data = seq(from = as.Date("2005-03-01"), to = as.Date("2005-04-01"), 
                               length.out = 31*24), Ora = rep(0:23, 31*24))
test <- test[0:744,]









final <- cbind(test, best_arima, best_ucm, best_ml)               #, best_ml



#df = subset(final, select = c("Data", "Ora", "Point Forecast", "fit") )     #, "ML"

df <- final
df$ARIMA <- round(as.numeric(df$best_arima))
df$UCM <- round(as.numeric(df$best_ucm))
df$ML <- round(as.numeric(df$best_ml))

df_backup <- df

keeps <- c("Data","Ora", "ARIMA", "UCM", "ML")
df <- df[keeps]




colnames(df) <- c("Date", "Hour", "ARIMA", "UCM","ML")      #, "ML"

df[df == "2005-04-01"] <- "2005-03-31"

write.csv2(df, "C:\\Users\\emanu\\Downloads\\812503_20220117.csv", row.names = F) 







#C:\Users\emanu\Downloads\Telegram Desktop



































