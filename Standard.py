import imp
from msilib.schema import Condition
from tokenize import String
from turtle import pu
import xmltodict
import pandas as pd
from sympy import *
import imp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
with open('defaulExposureStandardModel.xml',encoding='utf8') as fd:
    doc = xmltodict.parse(fd.read())
import time



class Standard():
    def __init__(self,standard):
        self.standard = standard
        self.public = None
        self.occupational = None
def getCode6s():
    data = {
        'MaxFrequency'  :[20             ,48             ,300                   ,6000                 ,150000       ,300000],
        'MinFrequency'  :[10             ,20             ,48                    ,300                  ,6000         ,150000],
        'Expression'    :["2"            ,"8.944e3/(f**0.5)","1.291"               ,"0.02619*(f*e-6)**0.6834" ,"10"         ,"6.67*f*(10**-5)/(10**6)"]
    }
    public = pd.DataFrame(data)
    data = {
        'MaxFrequency'  :[20             ,48             ,100            ,6000                 ,150000       ,300000],
        'MinFrequency'  :[10             ,20             ,48             ,100                  ,6000         ,150000],
        'Expression'    :["10"           ,"44.72/((f*10**-6)**0.5)" ,"6.455"          ,"0.6455*(f*(10**-6)**0.5)"      ,"50"           ,"3.33*(10**-4)*f*(10**-6)"]
    }
    occupational = pd.DataFrame(data)
    temp = Standard('Code6s')
    temp.public = public
    temp.occupational = occupational
    return temp

def getFCCs():
    data = {
        'MaxFrequency'         :[1.34        ,3                         ,30                     ,300         ,1500         ,100000],
        'MinFrequency'         :[0.3         ,1.34                      ,3                      ,30          ,300          ,1500],
        'Expression'           :["1000"      ,"1800/((f*10**-6)**2)"    ,"1800/((f*10**-6)**2)" ,"2"         ,"f/150e6"       ,"10"    ],
    }
    public = pd.DataFrame(data)
    data = {
        'MaxFrequency'         :[1.34        ,3                         ,30                     ,300         ,1500         ,100000],
        'MinFrequency'         :[0.3         ,1.34                      ,3                      ,30          ,300          ,1500],
        'Expression'           :["1000"        ,"1000"         ,"9000/((f*10**-6)**2)"   ,"10"          ,"f*10**-6/30"         ,"50"  ]
    }
    occupational = pd.DataFrame(data)
    temp = Standard('FCCs')
    temp.public = public
    temp.occupational = occupational
    return temp





def start():
    stand = []
    for standard in doc['ExposureStandardModel']['ExposureStandard']:
        st = Standard(standard = standard['Label'])
        if st.standard != "Serbian":
            data = []
            for refL in standard['ReferenceLevel']:
                for row in refL['FrequencyRange']:
                    data.append({
                        "MinFrequency":row["MinFrequency"],
                        "MaxFrequency":row["MaxFrequency"],
                        "Expression":row["Expression"]
                    })
                if (refL['Label'] == "Public") or (refL['Label'] == "General Population") or (refL['Label'] == "Uncontrolled Environments"):
                    st.public = pd.DataFrame(data)
                    st.public["MinFrequency"] = st.public["MinFrequency"].astype(float)
                    st.public["MaxFrequency"] = st.public["MaxFrequency"].astype(float)
                    st.public["Expression"] = st.public["Expression"].astype(str)
                    st.public["MinFrequency"] = st.public["MinFrequency"]*(10**-6)
                    st.public["MaxFrequency"] = st.public["MaxFrequency"]*(10**-6)
                else :
                    st.occupational = pd.DataFrame(data)
                    st.occupational["MinFrequency"] = st.occupational["MinFrequency"].astype(float)
                    st.occupational["MaxFrequency"] = st.occupational["MaxFrequency"].astype(float)
                    st.occupational["MinFrequency"] = st.occupational["MinFrequency"]*(10**-6)
                    st.occupational["MaxFrequency"] = st.occupational["MaxFrequency"]*(10**-6)
                    st.occupational["Expression"] = st.occupational["Expression"].astype(str)

                data = []
            stand.append(st)
    stand.append(getFCCs())
    stand.append(getCode6s())
    return stand
all = start()


def getStandard(standard = 'FCC'):
    if standard == 'ICNIRP':
        return all[0]
    elif standard == 'ARPANSA':
        return all[1]
    elif standard == 'FCC':
        return all[2]
    elif standard == 'BGVB11':
        return all[3]
    elif standard == 'Code6':
        return all[4]
    elif standard == 'FCCs':
        return all[5]
    elif standard == 'Code6s':
        return all[6]


def getZone(f,standard = 'FCC'):
        stnew = ''
        if standard == 'ICNIRP':
            stnew = all[0]
        elif standard == 'ARPANSA':
            stnew = all[1]
        elif standard == 'FCC':
            stnew = all[2]
        elif standard == 'BGVB11':
            stnew = all[3]
        elif standard == 'Code6':
            stnew = all[4]
        elif standard == 'FCCs':
            stnew = all[5]
        elif standard == 'Code6s':
            stnew = all[6]

        stnew.public = stnew.public.loc[(stnew.public['MinFrequency'] <= f) & (stnew.public['MaxFrequency'] > f)]
        stnew.occupational = stnew.occupational.loc[(stnew.occupational['MinFrequency'] <= f) & (stnew.occupational['MaxFrequency'] > f)]
        public = str(stnew.public['Expression'].to_numpy()[0])
        occupational = str(stnew.occupational['Expression'].to_numpy()[0])


        if 'f' in public:
            freq = symbols('f')
            gfg = sympify(public)
            public = gfg.subs(freq,f*10**6)
        else:
            public = float(public)
        
        if 'f' in occupational:
            freq = symbols('f')
            gfg = sympify(occupational)
            occupational = gfg.subs(freq,f*10**6)
        else:
            occupational = float(occupational)


        return public, occupational

