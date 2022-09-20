import pandas as pd

class Standard():
    def __init__(self,f,standard = 'FCC'):
        self.f = f
        self.standard = standard
        if standard == 'FCC':
            limits = 0
        elif standard == 'Code6':
            self.df = getCode6(f)

    def conditions(self,f=0):
        if f == 0:
            return self.df.loc[(self.df['Lower'] <= self.f) & (self.df['Upper'] > self.f)]
        else:
            return self.df.loc[(self.df['Lower'] <= f) & (self.df['Upper'] > f)]



def getCode6(f):
    data = {
        'Upper'         :[20             ,48             ,100            ,300                ,6000               ,15000      ,150000 ,300000],
        'Lower'         :[10             ,20             ,48             ,100                ,300                ,6000       ,15000  ,150000],
        'General Public':[2              ,8.944/(f**0.5) ,1.291          ,1.291              ,0.02619*(f**0.6834),10         ,10     ,6.67*(10**-5)],
        'Occupational'  :[10             ,44.72/(f**0.5) ,6.455          ,0.6455*(f**0.5)    ,0.6455*(f**0.5)    ,50         ,50     ,3.33*f*(10**-4)]
    }
    return pd.DataFrame(data)


#C6 = Standard(f=900,standard='Code6')

#print(C6.df.loc[(C6.df['Lower'] <= C6.f) & (C6.df['Upper'] > C6.f)])