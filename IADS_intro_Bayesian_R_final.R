
library(coda)
library(HDInterval)
library(LearnBayes)
library(MCMCpack)
library(rjags)

###########
#Example 1
###########

theta<-seq(0.01,0.99,by=0.01)
y<-c(1,1,1,1,1,0);n<-length(y)
a<-2;b<-2 ##play around with different values for different prior
prior.ex1<-dbeta(theta,shape1=a,shape2=b)
normalised.likelihood.ex1<-dbeta(theta,shape1=(sum(y)+1),shape2=(n-sum(y)+1))
posterior.ex1<-dbeta(theta,shape1=(a+sum(y)),shape2=(b+n-sum(y)))

plot(theta,normalised.likelihood.ex1,type="l",ylab="Density",ylim=c(0,max(c(normalised.likelihood.ex1,prior.ex1,posterior.ex1))))
lines(theta,prior.ex1,type="l",col=2)
lines(theta,posterior.ex1,type="l",col=3)
legend("topleft",c("Normalised Likelihood","Prior","Posterior"),lty=1,col=c(1,2,3))

#if we had more data
y<-rep(c(1,1,1,1,1,0),10);n<-length(y)
normalised.likelihood.ex1<-dbeta(theta,shape1=(sum(y)+1),shape2=(n-sum(y)+1))
posterior.ex1<-dbeta(theta,shape1=(a+sum(y)),shape2=(b+n-sum(y)))

plot(theta,normalised.likelihood.ex1,type="l",ylab="Density",ylim=c(0,max(c(normalised.likelihood.ex1,prior.ex1,posterior.ex1))))
lines(theta,prior.ex1,type="l",col=2)
lines(theta,posterior.ex1,type="l",col=3)
legend("topleft",c("Normalised Likelihood","Prior","Posterior"),lty=1,col=c(1,2,3))


###########
#Example 2
###########

theta<-seq(-10,10,by=0.01)
ybar<-6;n<-1;sigma2<-1
mu<-2;tau2<-1 ##play around with different values for different prior
prior.ex2<-dnorm(theta,mean=mu,sd=sqrt(tau2))
normalised.likelihood.ex2<-dnorm(theta,mean=ybar,sd=sqrt(sigma2/n))
credibility.weight<-(sigma2/n)/(sigma2/n+tau2)
posterior.ex2<-dnorm(theta,mean=(credibility.weight*mu+(1-credibility.weight)*ybar),sd=sqrt(1/(n/sigma2+1/tau2)))

plot(theta,normalised.likelihood.ex2,type="l",ylim=c(0,max(c(normalised.likelihood.ex2,prior.ex2,posterior.ex2))))
lines(theta,prior.ex2,type="l",col=2)
lines(theta,posterior.ex2,type="l",col=3)
legend("topleft",c("Normalised Likelihood","Prior","Posterior"),lty=1,col=c(1,2,3))

#if we had more data
n<-10
normalised.likelihood.ex2<-dnorm(theta,mean=ybar,sd=sqrt(sigma2/n))
credibility.weight<-(sigma2/n)/(sigma2/n+tau2)
posterior.ex2<-dnorm(theta,mean=(credibility.weight*mu+(1-credibility.weight)*ybar),sd=sqrt(1/(n/sigma2+1/tau2)))
plot(theta,normalised.likelihood.ex2,type="l",ylim=c(0,max(c(normalised.likelihood.ex2,prior.ex2,posterior.ex2))))
lines(theta,prior.ex2,type="l",col=2)
lines(theta,posterior.ex2,type="l",col=3)
legend("topleft",c("Normalised Likelihood","Prior","Posterior"),lty=1,col=c(1,2,3))


###############
##Point/interval/density estimation
##############

#Posterior mean
post.mean.ex1<-(a+sum(y))/(a+b+n)
plot(theta,posterior.ex1,type="l",col=3,ylim=c(0,3.2))
points(post.mean.ex1,dbeta(post.mean.ex1,shape1=(a+sum(y)),shape2=(b+n-sum(y))),pch=4)
legend(locator(1),legend="Mean",bty="n")

#Posterior mode
post.mode.ex1<-(a+sum(y)-1)/(a+b+n-2)
points(post.mode.ex1,dbeta(post.mode.ex1,shape1=(a+sum(y)),shape2=(b+n-sum(y))),pch=20)
legend(locator(1),legend="Mode",bty="n")

#Posterior median
post.median.ex1<-qbeta(0.50,shape1=(a+sum(y)),shape2=(b+n-sum(y)))
points(post.median.ex1,dbeta(post.median.ex1,shape1=(a+sum(y)),shape2=(b+n-sum(y))),pch=5)
legend(locator(1),legend="Median",bty="n")

#Suppose we want to find E(h(theta)|y), where h(theta)=log(theta/(1-theta))
expected.h.fn<-function(theta,shape1,shape2){
log(theta/(1-theta))*dbeta(theta,shape1=shape1,shape2=shape2)
}

#numerical integration
E_h_theta<-integrate(f=expected.h.fn,lower=0,upper=1,shape1=(a+sum(y)),shape2=(b+n-sum(y)))$value #true value=0.95

##Interval estimation
#95% equal-tailed interval
post.equal.interval.ex1<-c(qbeta(0.025,shape1=(a+sum(y)),shape2=(b+n-sum(y))),qbeta(0.975,shape1=(a+sum(y)),shape2=(b+n-sum(y))))
abline(v=(post.equal.interval.ex1),lty=3)

#95% HPD interval
post.HPD.interval.ex1<-hdi(qbeta,credMass=0.95,shape1=(a+sum(y)),shape2=(b+n-sum(y)))
abline(v=(post.HPD.interval.ex1),lty=5,col="blue");abline(h=dbeta(post.HPD.interval.ex1[1],shape1=(a+sum(y)),shape2=(b+n-sum(y))),lty=5,col="blue")

##################################################################
##Assume you do not know the posterior density, but can generate independent samples. Repeat the above and compare with the true value.
################################################################

#Posterior samples (independent samples)
post.sample.ex1<-rbeta(10000,shape1=(a+sum(y)),shape2=(b+n-sum(y)))

#mean
post.sample.mean.ex1<-mean(post.sample.ex1)
rbind(c("True","Sample"),c(post.mean.ex1,post.sample.mean.ex1))

#mode
estimate_mode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}
post.sample.mode.ex1<-estimate_mode(post.sample.ex1)
rbind(c("True","Sample"),c(post.mode.ex1,post.sample.mode.ex1))

#median
post.sample.median.ex1<-median(post.sample.ex1)
rbind(c("True","Sample"),c(post.median.ex1,post.sample.median.ex1))


#Suppose we want to find E(h(theta)|y), where h(theta)=log(theta/(1-theta))
E_h_theta_MC<-mean(log(post.sample.ex1/(1-post.sample.ex1))) #MC estimate 
rbind(c("True","Sample"),c(E_h_theta,E_h_theta_MC)) #close to the true value!

##Interval estimation
#95% equal-tailed interval
post.equal.interval.sample.ex1<-c(quantile(post.sample.ex1,0.025),quantile(post.sample.ex1,0.975))
rbind(c("True","Sample"),cbind(post.equal.interval.ex1,post.equal.interval.sample.ex1))

#95% HPD interval
post.HPD.interval.sample.ex1<-HPDinterval(as.mcmc(post.sample.ex1),prob=0.95)

##Density estimation
plot(theta,posterior.ex1,type="l",col=3,ylim=c(0,3.2))
lines(density(post.sample.ex1),col=2);legend("topleft",c("True","Sample"),lty=1,col=c(3,2))

abline(v=(post.equal.interval.sample.ex1),lty=3)
abline(v=(post.HPD.interval.sample.ex1),lty=5,col="blue");abline(h=dbeta(post.HPD.interval.sample.ex1[1],shape1=(a+sum(y)),shape2=(b+n-sum(y))),lty=5,col="blue")



#######################
#######################
##MCMC methods
#######################
#######################

n<-25;ybar<-0.05;prior.location=0
theta<-seq(-3,3,by=0.001)
normalised.likelihood<-dnorm(theta,mean=ybar,sd=sqrt(1/n))
cauchy.prior<-dcauchy(theta,location=prior.location)
plot(theta,normalised.likelihood,type="l",col=1)
lines(theta,cauchy.prior,col=2) 
legend("topright",c("Normalised Likelihood","Cauchy Prior"),lty=1,col=c(1,2))

#Independence sampler

IS.MCMC<-function(m,initial,n,ybar,prior.location){
post.sample<-vector(length=m)
theta.current<-initial
accept<-0
for (i in 1:m){
theta.propose<-rcauchy(1,location=prior.location)
alpha.log<-dnorm(theta.propose,mean=ybar,sd=sqrt(1/n),log=TRUE)-dnorm(theta.current,mean=ybar,sd=sqrt(1/n),log=TRUE) #ratio of likelihoods
if (log(runif(1))<alpha.log) {theta.current<-theta.propose
accept<-accept+1
}
post.sample[i]<-theta.current
}
list(post.sample=post.sample,accept.rate=accept/m)
}

m<-101000;initial<-5
post.sample.IS.MCMC<-IS.MCMC(m=m,initial=initial,n=n,ybar=ybar,prior.location=prior.location)

##Some basic MCMC diagnostic checks (manually)

post.sample.IS.MCMC$accept.rate #check acceptance rate
plot(post.sample.IS.MCMC$post.sample,type="l") #some burn-in required
acf(post.sample.IS.MCMC$post.sample) #can consider thinning

#Burn-in
burnin<-1000
post.sample.IS.MCMC.final<-post.sample.IS.MCMC$post.sample[-(1:burnin)]

#Thinning
thinning<-10
post.sample.IS.MCMC.final<-post.sample.IS.MCMC.final[seq(1,m-burnin,by=thinning)] #thin by 10

plot(post.sample.IS.MCMC.final,type="l") 
acf(post.sample.IS.MCMC.final) 

mean(post.sample.IS.MCMC.final)
var(post.sample.IS.MCMC.final)
plot(theta,normalised.likelihood,type="l",col=1);lines(theta,cauchy.prior,col=2)
lines(density(post.sample.IS.MCMC.final),col=3)
legend("topright",c("Normalised Likelihood","Cauchy Prior","Posterior"),lty=1,col=c(1,2,3))



#Random-walk Metropolis Hastings

RW.MCMC<-function(m,initial,sigma2.prop,n,ybar,prior.location){
post.sample<-vector(length=m)
theta.current<-initial
accept<-0
for (i in 1:m){
theta.propose<-rnorm(1,theta.current,sd=sqrt(sigma2.prop))
alpha.log<-dnorm(theta.propose,mean=ybar,sd=sqrt(1/n),log=TRUE)+dcauchy(theta.propose,location=prior.location,log=TRUE)-dnorm(theta.current,mean=ybar,sd=sqrt(1/n),log=TRUE)-dcauchy(theta.current,location=prior.location,log=TRUE) #ratio of posteriors
if (log(runif(1))<alpha.log) {theta.current<-theta.propose
accept<-accept+1
}
post.sample[i]<-theta.current
}
list(post.sample=post.sample,accept.rate=accept/m)
}

m<-10000;initial<-5;sigma2.prop<-1 #change sigma2.prop to see the resulting trajectories
post.sample.RW.MCMC<-RW.MCMC(m=m,initial=initial,sigma2.prop=sigma2.prop,n=n,ybar=ybar,prior.location=prior.location)

post.sample.RW.MCMC$accept.rate #check acceptance rate
plot(post.sample.RW.MCMC$post.sample,type="l") #some burn-in required
acf(post.sample.RW.MCMC$post.sample) #can consider thinning

# OR using LearnBayes
logpost.RW.MCMC<-function(theta,n,ybar,prior.location) {dnorm(theta,mean=ybar,sd=sqrt(1/n),log=TRUE)+dcauchy(theta,location=prior.location,log=TRUE)}
scale<-0.85
post.sample.RW.MCMC.LB<-rwmetrop(logpost.RW.MCMC,proposal=list(var=1,scale=scale),start=5,m=10000,n=n,ybar=ybar,prior.location=prior.location)

post.sample.RW.MCMC.LB$accept #tune scale until 0.15-0.40
post.sample.RW.MCMC.LB<-mcmc(post.sample.RW.MCMC.LB$par)
plot(post.sample.RW.MCMC.LB,type="l") 
acf(post.sample.RW.MCMC.LB)

##Some basic MCMC diagnostic checks (can do manually as before OR use R package "coda")

#Assume we generate 4 parallel chains with different initial values

post.sample.RW.MCMC.2<-RW.MCMC(m=m,initial=10,sigma2.prop=sigma2.prop,n=n,ybar=ybar,prior.location=prior.location)
post.sample.RW.MCMC.3<-RW.MCMC(m=m,initial=-5,sigma2.prop=sigma2.prop,n=n,ybar=ybar,prior.location=prior.location)
post.sample.RW.MCMC.4<-RW.MCMC(m=m,initial=-10,sigma2.prop=sigma2.prop,n=n,ybar=ybar,prior.location=prior.location)

#To use coda diagnostic tools, need to declare as "mcmc" objects using mcmc or as.mcmc
post.sample.RW.MCMC.object<-mcmc(cbind(post.sample.RW.MCMC$post.sample,post.sample.RW.MCMC.2$post.sample,post.sample.RW.MCMC.3$post.sample,post.sample.RW.MCMC.4$post.sample),start=1,thin=1)

summary(post.sample.RW.MCMC.object)
plot(post.sample.RW.MCMC.object)
acfplot(post.sample.RW.MCMC.object)
autocorr(post.sample.RW.MCMC.object)
densplot(post.sample.RW.MCMC.object,col=3)
lines(theta,normalised.likelihood,type="l",col=1);lines(theta,cauchy.prior,col=2);legend("topright",c("Normalised Likelihood","Cauchy Prior","Posterior"),lty=1,col=c(1,2,3))

effectiveSize(post.sample.RW.MCMC.object)

1-pnorm(geweke.diag(post.sample.RW.MCMC.object)$z) #reject is <0.025 for 5% test
geweke.plot(post.sample.RW.MCMC.object)

post.sample.RW.MCMC.mcmclist<-mcmc.list(mcmc(post.sample.RW.MCMC$post.sample),mcmc(post.sample.RW.MCMC.2$post.sample),mcmc(post.sample.RW.MCMC.3$post.sample),mcmc(post.sample.RW.MCMC.4$post.sample))
gelman.diag(post.sample.RW.MCMC.mcmclist)
gelman.plot(post.sample.RW.MCMC.mcmclist)


###########################
##Practical Example 1
###########################

dose<-rep(c(1,2,4,8,16,32),2)
ldose<-log(dose,base=2)
numdead<-c(1,4,9,13,18,20,0,2,6,10,12,16)
sex<-factor(rep(c("M","F"),c(6,6)))
SF<-cbind(numdead,numalive=20-numdead)
resp<-rep(rep(c(1,0),12),times=t(SF))
budworm<-data.frame(resp,ldose=rep(ldose,each=20),sex=rep(sex,each=20))

#Exploratory analysis

plot(dose[1:6],numdead[1:6]/20,ylab="prob",ylim=c(0,1),xlab="dose",log="x",pch="M")
points(dose[7:12],numdead[7:12]/20,pch="F")
lines(dose[1:6],fit.classical2$fit[1:6],lty=1);lines(dose[7:12],fit.classical2$fit[7:12],lty=2)

##Classical GLM logistic regression

fit.classical<-glm(resp~sex*ldose,data=budworm,family=binomial(link="logit"))
summary(fit.classical)

w<-rep(20,12)
fit.classical2<-glm(numdead/w~sex*ldose,weights=w,family=binomial)
summary(fit.classical2)

##Bayesian inference

#Method 1 (MCMClogit)

fit<-MCMClogit(resp~sex*ldose,data=budworm)
summary(fit)
effectiveSize(fit)
plot(fit)
acfplot(fit) #suggests thinning
fit.thin<-MCMClogit(resp~sex*ldose,data=budworm,mcmc=200000,thin=20)
summary(fit.thin)
plot(fit.thin)
acfplot(fit.thin) #much better
HPDinterval(fit.thin)

ld50F<-as.mcmc(2^(-fit.thin[,1]/fit.thin[,3]))
ld50M<-as.mcmc(2^(-(fit.thin[,1]+fit.thin[,2])/(fit.thin[,3]+fit.thin[,4])))
ld50<-mcmc(cbind(M=ld50M,F=ld50F))

ld50F<-ifelse(fit.thin[,3]>0,2^(-fit.thin[,1]/fit.thin[,3]),ifelse(fit.thin[,1]>0,0,Inf))

abline(h=0.5)
abline(v=mean(ld50M),lty=1);abline(v=mean(ld50F),lty=2)

#Method 2 (LearnBayes)

X<-model.matrix(fit.classical2)
logpost<-function(beta){
sum(dbinom(numdead,w,plogis(X%*%drop(beta)),log=TRUE))+sum(dnorm(beta,mean=0,sd=sqrt(1000),log=TRUE))}

scale<-0.95
fit2<-rwmetrop(logpost,list(var=vcov(fit.classical2),scale=scale),coef(fit.classical2),m=10000)
fit2$accept #tune to 0.15-0.40

fit2<-mcmc(fit2$par)
summary(fit2)
plot(fit2)
effectiveSize(fit2)
acfplot(fit2) #again, may consider thinning

#Method 3 (JAGS model)

data<-list(numdead=numdead,ldose=ldose)
inits<-list(list(alphaM=0,betaM=0,alphaF=0,betaF=0))
vars<-c("alphaM","alphaF","betaM","betaF")
budworm.jags<-jags.model("G:/Teaching@essex/IADS summer school/budworm.jags",data=data,inits=inits,n.chains=1,n.adapt=500)
fit3<-coda.samples(budworm.jags,vars,n.iter=10000)
summary(fit3)
plot(fit3)
acfplot(fit3) #again, may consider thinning
effectiveSize(fit3)

###########################
##Practical Example 2
###########################

puffin #data

#Assuming Gaussian distribution

##Classical GLM linear regression
fit.puffin.classicalG<-glm(Nest~Grass+Soil+Angle+Distance,family=gaussian,data=puffin)
summary(fit.puffin.classicalG)

##Method 1(MCMCregress)

fit.puffin<-MCMCregress(Nest~Grass+Soil+Angle+Distance,data=puffin,burnin=1000,mcmc=100000,thin=10)
summary(fit.puffin)
plot(fit.puffin)
acfplot(fit.puffin)

##Method 3 (JAGS model)

X<-model.matrix(~Grass+Soil+Angle+Distance,data=puffin)
data<-list(Nest=puffin$Nest,X=X)
inits<-list(list(beta=rep(0,5),sigma=1))
vars<-c("beta","sigma")
puffin.jags<-jags.model("G:/Teaching@essex/IADS summer school/puffin.jags",data=data,inits=inits,n.chain=1)
fit3.puffin<-coda.samples(puffin.jags,vars,n.iter=200000,thin=20)

summary(fit3.puffin)
plot(fit3.puffin)
acfplot(fit3.puffin) #consider thinning
effectiveSize(fit3.puffin)


#Assuming Poisson distribution

##Classical GLM Poisson regression
fit.puffin.classicalP<-glm(Nest~Grass+Soil+Angle+Distance,family=poisson,data=puffin)
summary(fit.puffin.classicalP)

##Method 1(MCMCpoisson)

fit.puffinP<-MCMCpoisson(Nest~Grass+Soil+Angle+Distance,data=puffin,burnin=1000,mcmc=100000,thin=10)
summary(fit.puffinP)
plot(fit.puffinP)
acfplot(fit.puffinP)

##Method 2 (LearnBayes)

logpost.puffinP<-function(beta) {sum(dpois(puffin$Nest,lambda=exp(X%*%drop(beta)),log=TRUE))+sum(dnorm(beta,mean=0,sd=10,log=TRUE))}
scale<-0.85
fit2.puffinP<-rwmetrop(logpost.puffinP,list(var=vcov(fit.puffin.classicalP),scale=scale),coef(fit.puffin.classicalP),m=10000)

fit2.puffinP$accept #tune to 0.15-0.40

fit2.puffinP<-mcmc(fit2.puffinP$par)
summary(fit2.puffinP)
plot(fit2.puffinP)
acfplot(fit2.puffinP) #consider thinning
effectiveSize(fit2.puffinP)

##Method 3 (JAGS model)

X<-model.matrix(~Grass+Soil+Angle+Distance,data=puffin)
data<-list(Nest=puffin$Nest,X=X)
inits<-function() (list(beta=c(2,rep(0,4))))
vars<-c("beta")
puffinP.jags<-jags.model("G:/Teaching@essex/IADS summer school/puffinP.jags",data=data,inits=inits,n.chain=4)
fit3.puffinP<-coda.samples(puffinP.jags,vars,n.iter=100000,thin=10)

summary(fit3.puffinP)
plot(fit3.puffinP)
acfplot(fit3.puffinP) #consider thinning
effectiveSize(fit3.puffinP)

#Assuming Negative-Binomial distribution

##Classical GLM NB regression
library(MASS)
fit.puffin.classicalNB<-glm.nb(Nest~Grass+Soil+Angle+Distance,maxit=50,data=puffin)
summary(fit.puffin.classicalNB)

##Method 1(MCMCnegbin)

fit.puffinNB<-MCMCnegbin(Nest~Grass+Soil+Angle+Distance,data=puffin,burnin=1000,mcmc=10000,thin=1)
summary(fit.puffinNB)
plot(fit.puffinNB)
acfplot(fit.puffinNB)

##Method 3 (JAGS model)

X<-model.matrix(~Grass+Soil+Angle+Distance,data=puffin)
data<-list(Nest=puffin$Nest,X=X)
inits<-function() (list(beta=c(2,rep(0,4))))
vars<-c("beta")
puffinNB.jags<-jags.model("G:/Teaching@essex/IADS summer school/puffinNB.jags",data=data,inits=inits,n.chain=1)
fit3.puffinNB<-coda.samples(puffinNB.jags,vars,n.iter=100000,thin=10)

summary(fit3.puffinNB)
plot(fit3.puffinNB)
acfplot(fit3.puffinNB) #consider thinning
effectiveSize(fit3.puffinNB)

puffinNB3.jags<-jags.model("G:/Teaching@essex/IADS summer school/puffinNB2.jags",data=data,inits=inits,n.chain=1)
fit3.puffinNB2<-coda.samples(puffinNB2.jags,vars,n.iter=10000,thin=1)

summary(fit3.puffinNB2)
plot(fit3.puffinNB2)
acfplot(fit3.puffinNB2) #consider thinning
effectiveSize(fit3.puffinNB2)

