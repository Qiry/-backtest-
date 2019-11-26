Code for backtesting.
steps:
1. 
def initialize():
    glob.spec()
    glob.data
    glob.date
    glob.startdate
    glob.enddate
    glob.benchmark
    
2.
def strategy():
    order(glob,securityname/code,amt,'buy'/'sell')

3.
bkt.backtest(initialize,glob,strategy)
