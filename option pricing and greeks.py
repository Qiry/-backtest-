# coding: utf-8

from scipy.stats import norm
from math import *

#T in years, v in %, r in %

class optionpricing():
    def __init__(self,S,X,T,r,v,q=0):
        self.spot=S
        self.strike=X
        self.time=T
        self.riskfree=r/100
        self.vol=v/100
        self.q=q
        self.d1=(log(self.spot/self.strike)+((self.riskfree-self.q)+self.vol*self.vol/2)*self.time)/(self.vol*sqrt(self.time))
        self.d2=self.d1-self.vol*sqrt(self.time)
    def callprice(self):
        print(self.d1,norm.cdf(self.d1))
        print(self.d2)
        return self.spot*exp(-self.q*self.time)*norm.cdf(self.d1)-self.strike*exp(-self.riskfree*self.time)*norm.cdf(self.d2)
        
    def putprice(self):
        return self.strike*exp(-self.riskfree*self.time)*norm.cdf(-self.d2)-self.spot*exp(-self.q*self.time)*norm.cdf(-self.d1)
        


