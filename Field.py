
from matplotlib import pyplot as plt
import numpy as np
from numpy import *
import pandas as pd
import pandas as pd
import mayavi.mlab as mlab
from Standard import getZone



import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
    
class Field:
    def __init__(self,df,f,type = 'Feko',standard = 'FCC',restriction = True):
        self.standard = standard
        self.df = df
        self.f = f

        if restriction:
            maxFreq = getZone(self.f,standard)[1]
            minFreq = getZone(self.f,standard)[0]  
            self.df['Restriction'] = 1
            self.df.loc[minFreq > self.df['S'],'Restriction'] = 0
            self.df.loc[maxFreq < self.df['S'],'Restriction'] = 2

class Fekofield(Field):
    def __init__(self,power,source,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df):
        super().__init__(df,frequency*(10**-6),type = 'Feko',standard=standard, restriction = False)
        self.source = source
        self.frequency = frequency
        self.coordSystem= coordSystem
        self.xSamples = xSamples
        self.ySamples = ySamples
        self.zSamples = zSamples
        self.power = power

        i = 0
        if xSamples > 1: i+=1
        if ySamples > 1: i+=1
        if zSamples > 1: i+=1
        self.dimentions = i
        if self.dimentions == 2:
            if xSamples > 1 and ySamples > 1:
                self.axis = ['X','Y'] 
            elif xSamples > 1 and zSamples > 1:
                self.axis = ['X','Z']
            elif ySamples > 1 and zSamples > 1:
                self.axis = ['Y','Z']
    
    def plot2D(self, c='S', color='Reds',show = True,method = 'matplotlib'):
        if method == 'matplotlib':
            fig, ax = plt.subplots(1)
            ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[c],cmap = color)
            plt.colorbar(ax1)
            ax.set_xlabel(self.axis[0])
            ax.set_ylabel(self.axis[1])
            ax.set_title("{} over {}{} plane".format(c,self.axis[0],self.axis[1]))
            if show:
                plt.show()
        elif method == "mayavi":
            self.df = self.df.sort_values(by=['Z','Y','X'])
            arr = self.df[c].to_numpy()
            arr = arr.reshape(self.xSamples,self.ySamples)
            mlab.surf(arr,warp_scale = 'auto')
            mlab.show()




      
def GetField(filenameE,filenameH,S = 'S(E)',compress = False ,standard = 'FCC',power = 80):
    source= ''
    frequency= 900
    coordSystem= ''
    xSamples= 0
    ySamples= 0
    zSamples= 0
    global i
    i = 0
    filenameE = 'venv/Include/CADFeko/{}'.format(filenameE)
    filenameH = 'venv/Include/CADFeko/{}'.format(filenameH)

    with open(filenameE, 'r') as file:
        for line in file:
            if '##Source: ' in line:
                source = line[:-1].split("##Source: ",1)[1]
            elif "#Frequency: " in line:
                frequency = int(float(line[:-1].split("#Frequency:   ",1)[1]))  
            elif "#Coordinate System: " in line:
                coordSystem = line[:-1].split("#Coordinate System: ",1)[1]
            elif "#No. of X Samples: " in line:
                xSamples = int(line[:-1].split("No. of X Samples: ",1)[1])
            elif "#No. of Y Samples: " in line:
                ySamples = int(line[:-1].split("No. of Y Samples: ",1)[1])
            elif "#No. of Z Samples: " in line:
                zSamples = int(line[:-1].split("No. of Z Samples: ",1)[1])

                global dataT
                dataT = np.zeros((ySamples*xSamples*zSamples,9)) 
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataT[i] = line[4:-1].split('   ')
                i+=1
        df = pd.DataFrame(dataT,columns=['X','Y','Z','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)'])
        df = df.astype(float)
        df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        df['phi'] = np.arctan(df['Y']/df['X'])
        df['theta'] = np.arctan(df['X']/df['Z'])
        dfG = GetFarField('IEC-62232-panel-antenna (4)_FarField1.ffe')
        phi = df['phi']
        theta = df['theta']
        df['phi'] = np.abs(np.round(df['phi']*180/np.pi))
        df['theta'] = np.abs(np.round(df['theta']*180/np.pi))
        df = df.merge(dfG,how='left',on=['phi','theta'])
        df['phi'] = phi
        df['theta'] = theta
    file.close()

    with open(filenameH, 'r') as file:
        dataH = np.zeros((ySamples*xSamples*zSamples,6))
        i = 0
        for line in file:
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataH[i] = line[4:-1].split('   ')[3:]
                i+=1
    file.close()
    df['Re(Hx)'] = dataH[:,0]
    df['Im(Hx)'] = dataH[:,1]
    df['Re(Hy)'] = dataH[:,2]
    df['Im(Hy)'] = dataH[:,3]
    df['Re(Hz)'] = dataH[:,4]
    df['Im(Hz)'] = dataH[:,5]

    df['Ex'] = (df['Re(Ex)'] + df['Im(Ex)']*1j)/np.sqrt(2)
    df['|Ex|'] = np.absolute(df['Ex'])
    df['Ey'] = (df['Re(Ey)'] + df['Im(Ey)']*1j)/np.sqrt(2)
    df['|Ey|'] = np.absolute(df['Ey'])
    df['Ez'] = (df['Re(Ez)'] + df['Im(Ez)']*1j)/np.sqrt(2)
    df['|Ez|'] = np.absolute(df['Ez'])
    df['Hx'] = (df['Re(Hx)'] + df['Im(Hx)']*1j)/np.sqrt(2)
    df['Hy'] = (df['Re(Hy)'] + df['Im(Hy)']*1j)/np.sqrt(2)
    df['Hz'] = (df['Re(Hz)'] + df['Im(Hz)']*1j)/np.sqrt(2)

    df['|E|'] = np.sqrt(np.absolute(df['Ex'])**2+ np.absolute(df['Ey'])**2 + np.absolute(df['Ez'])**2)

    #df['Hx'] = np.conj(df['Hx'])
    #df['Hy'] = np.conj(df['Hy'])
    #df['Hz'] = np.conj(df['Hz'])
    df['Sx'] = df['Ey']*df['Hz'] - df['Ez']*df['Hy']
    df['Sy'] = df['Ez']*df['Hx'] - df['Ex']*df['Hz']
    df['Sz'] = df['Ex']*df['Hy'] - df['Ey']*df['Hx']

    df['Full wave'] = np.sqrt(np.absolute(df['Sx'])**2 + np.absolute(df['Sy'])**2 + np.absolute(df['Sz'])**2)
    df['Classical'] = Classical(df['|E|'].to_numpy())
    df['OET65'] = OET65mesh(df['R'],df['Gain'])
    df['IEC Peak'] = IECmeshPeakSector(df['R'], df['phi'], df['theta'])
    df['IEC Average'] = IECmeshAverageSector(df['R'], df['phi'], df['theta'])
    df['EMSS Peak'] = EMSSmeshPeakSector(df['R'], df['phi'], df['theta'])
    df['EMSS Average'] = EMSSmeshAverageSector(df['R'], df['phi'], df['theta'])

    df['S'] = df['Full wave']
    if compress:
        df = df.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|','Full wave'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)

   

    def PowerAtPoint(data):
        S = np.zeros(len(data))
        for j in range(len(data)):
            for i in range(3,9):
                S[j] += data[j][i]**2
        S = S/(2*377) 
        return(S)

    def plotPowerLine(line,axis):
        plt.plot(line[axis],line['S'])
        plt.show() 

    def getS(self):
        dfS = self.df[['X','Y','Z','S']].copy()
        dfS = dfS.astype(float)
        return dfS

    def GetE(self):
        dfE = self.df[['X','Y','Z','|E|']].copy()
        dfE =  dfE.astype(float)
        return dfE

    def GetH(self):
        dfH = self.df[['X','Y','Z','|H|']].copy()
        dfH =  dfH.astype(float)
        return dfH

    def plot2DZones(self,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True):
        colors = {0:Ncolor,1:GPcolor,2:Ocolor}
        #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))

        groups = self.df.groupby('Restriction')
       
        fig, ax = plt.subplots(1, figsize=(xfig,yfig))
        for label, group in groups:
            ax.scatter(group[axis1], group[axis2], 
                    c=group['Restriction'].map(colors), label=label)

        ax.set(xlabel= axis1, ylabel=axis2)
        ax.set_title("Restricions with %d signal" % self.f)
        ax.legend(title='Restriction levels')
        if show:
            plt.show()



    def plot2D(self,c='S',color = 'Reds',method = 'matplotlib'):
        if method == 'matplotlib':
            fig, ax = plt.subplots(1)
            ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[c],cmap = color)
            plt.colorbar(ax1)
            ax.set_xlabel(self.axis[0])
            ax.set_ylabel(self.axis[1])
            ax.set_title("{} over {}{} plane".format(c,self.axis[0],self.axis[1]))


    def compareToSelf(self,standard1,S2 = 'S',S1 = 'S',standard2 = None,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True,c='Restriction'):
        if c == 'Restriction':
            colors = {0:Ncolor,1:GPcolor,2:Ocolor}

            if standard2 == None:
                standard2 = self.standard

            maxFreq1 = getZone(self.f,standard1)[1]
            minFreq1 = getZone(self.f,standard1)[0]  
            self.df['Restriction1'] = 1
            self.df.loc[minFreq1 > self.df[S1],'Restriction1'] = 0
            self.df.loc[maxFreq1 < self.df[S1],'Restriction1'] = 2


            maxFreq2 = getZone(self.f,standard2)[1]
            minFreq2 = getZone(self.f,standard2)[0]  
            self.df['Restriction2'] = 1
            self.df.loc[minFreq2 > self.df[S2],'Restriction2'] = 0
            self.df.loc[maxFreq2 < self.df[S2],'Restriction2'] = 2


            groups1 = self.df.groupby('Restriction1')
            groups2 = self.df.groupby('Restriction2')

        
            fig, (ax1,ax2) = plt.subplots(2,1)
            fig.set_size_inches(14.5,10.5)
            for label, group in groups1:
                ax1.scatter(group[axis1], group[axis2], 
                        c=group['Restriction1'].map(colors), label=label)

            for label, group in groups2:
                ax2.scatter(group[axis1], group[axis2], 
                        c=group['Restriction2'].map(colors), label=label)

            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("{}".format(standard1))
            ax1.legend(title='Restriction levels')
            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("{}".format(standard2))
            ax2.legend(title='Restriction levels')

            self.df = self.df.drop(columns = ['Restriction1','Restriction2'])
        if show:
            plt.show()


    def compareToSurface2D(self,field,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True,c='Restriction'):
        if c == 'Restriction':
            colors = {0:Ncolor,1:GPcolor,2:Ocolor}
            #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))
            groups1 = self.df.groupby('Restriction')
            groups2 = field.df.groupby('Restriction')

        
            fig, (ax1,ax2) = plt.subplots(2,1)
            for label, group in groups1:
                ax1.scatter(group[axis1], group[axis2], 
                        c=group['Restriction'].map(colors), label=label)

            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("Restricions with %d signal" % self.f)
            ax1.legend(title='Restriction levels')

            for label, group in groups2:
                ax2.scatter(group[axis1], group[axis2], 
                        c=group['Restriction'].map(colors), label=label)

            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("Restricions with %d signal" % self.f)
            ax2.legend(title='Restriction levels')
        else: 
            fig, (ax1,ax2) = plt.subplots(2,1)
            ax1.scatter(self.df[axis1], self.df[axis2], c=self.df[c])
            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("Restrictions with %d signal" % self.f)

            ax2.scatter(field.df[axis1], field.df[axis2], c=self.df[c])
            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("Restrictions with %d signal" % self.f)

        if show:
            plt.show()


def GetFarField(filename,compress = True,standard = 'FCC',power = 80):
    source= ''
    frequency= 900
    coordSystem= ''
    thetaSamples = 0
    phiSamples = 0
    global i
    i = 0
    filenameff = 'venv/Include/CADFeko/{}'.format(filename)

    with open(filenameff, 'r') as file:
        for line in file:
            if '##Source: ' in line:
                source = line[:-1].split("##Source: ",1)[1]
            elif "#Frequency: " in line:
                frequency = int(float(line[:-1].split("#Frequency:   ",1)[1]))  
            elif "#Coordinate System: " in line:
                coordSystem = line[:-1].split("#Coordinate System: ",1)[1]
            elif "#No. of Theta Samples: " in line:
                thetaSamples = int(line[:-1].split("#No. of Theta Samples: ",1)[1])
            elif "#No. of Phi Samples: " in line:
                phiSamples = int(line[:-1].split("#No. of Phi Samples: ",1)[1])
                global dataT
                dataT = np.zeros((thetaSamples*phiSamples,9)) 
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataT[i] = line[4:-1].split('   ')
                i+=1
        df = pd.DataFrame(dataT,columns=['theta','phi','Re(Etheta)','Im(Etheta)','Re(Ephi)','Im(Ephi)','Directivity(Theta)','Directivity(Phi)','Gain'])
        df = df.astype(float)
        df = df.drop(columns=['Re(Etheta)','Im(Etheta)','Re(Ephi)','Im(Ephi)','Directivity(Theta)','Directivity(Phi)'])
    file.close()

    #df['Etheta'] = (df['Re(Etheta)'] + df['Im(Etheta)']*1j)/np.sqrt(2)
    #df['Ephi'] = (df['Re(Ephi)'] + df['Im(Ephi)']*1j)/np.sqrt(2)
    #df['|E|'] = np.sqrt(np.absolute(df['Etheta'])**2 + np.absolute(df['Ephi'])**2)
    #df['S(E)'] = df['|E|']**2/(337*2)
    return df

def plotFarField(df):
    phi, theta  = mgrid[0:361:1,0:181:1]
    Gnum = 10**(df['Directivity(Total)'].to_numpy()/10)
    lamda = 1/3
    f = Gnum
    f = np.reshape(f,(361,181))
    x = f*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
    y = f*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
    z = f*np.cos(theta*np.pi/180)
    mlab.mesh(x, y, z)
    mlab.show()


def test_mesh(df,error = 1,S = 10):

    temp = df.loc[(df['S'] >= S-error) & (df['S'] < S+error)]
    temp = temp.sort_values(by = ['phi','theta','R'])

    #idx = temp.groupby(['theta','phi'])['S'].transform(max) == temp['S']
    #temp = temp.sort_values(by = ['theta','phi'])
    #temp = temp[idx]

    mlab.figure(bgcolor=(1, 1, 1))  # Make background white.
    mesh = mlab.mesh(temp['X'],temp['X'],temp['X'])
    #points3D = mlab.points3d(temp['X'],temp['Y'],temp['Z'])
    mlab.outline(color=(0, 0, 0))
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
    axes.title_text_property.color = (0.0, 0.0, 0.0)
    axes.title_text_property.font_family = 'times'
    axes.label_text_property.color = (0.0, 0.0, 0.0)
    axes.label_text_property.font_family = 'times'
    # mlab.savefig("vector_plot_in_3d.pdf")
    mlab.gcf().scene.parallel_projection = True  # Source: <<https://stackoverflow.com/a/32531283/2729627>>.
    mlab.orientation_axes()  # Source: <<https://stackoverflow.com/a/26036154/2729627>>.
    mlab.show()

#def plotSbyPhase
def plotSZones(df, *args,Y = 'y', X = 'x', error = 0.01,round = 3):
    for S in args:
        temp = df.loc[(df['S'] >= S-error) & (df['S'] < S+error)]
        temp['theta'] = np.round(temp['theta'],round)
        temp['phi'] = np.round(temp['phi'],round)
        idx = temp.groupby(['phi','theta'])['S'].transform(max) == temp['S']
        temp = temp.sort_values(by = ['phi'])
        plt.plot(temp[idx]['X'],temp[idx]['Y'], '-',label = 'Full wave = {}W/m'.format(S))

        temp = df.loc[(df['ICNIRP Peak'] >= S-error) & (df['ICNIRP Peak'] < S+error)]
        temp['theta'] = np.round(temp['theta'],round)
        temp['phi'] = np.round(temp['phi'],round)
        idx = temp.groupby(['phi','theta'])['ICNIRP Peak'].transform(max) == temp['ICNIRP Peak']
        plt.plot(temp[idx]['X'],temp[idx]['Y'],'o', label = 'ICNIRP = {}W/m'.format(S))

        temp = df.loc[(df['OET65'] >= S-error) & (df['OET65'] < S+error)]
        temp['theta'] = np.round(temp['theta'],round)
        temp['phi'] = np.round(temp['phi'],round)
        idx = temp.groupby(['phi','theta'])['OET65'].transform(max) == temp['OET65']
        plt.plot(temp[idx]['X'],temp[idx]['Y'], '+',label = 'OET 65 = {} W/m'.format(S))

    plt.title('Comparing various simulation methods of a 900Mhz,80W sector antenna at various S values')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.xlim([0,40])
    plt.ylim([-40,40])
    plt.show()


def plotByCartesian(df, *X,error = 0.1):
    for x in X:
        temp = df.loc[(df['X'] >= x-error) & (df['X'] < x+error)]
        idx = temp.groupby(['Z','Y'])['X'].transform(max) == temp['X']
        plt.plot(temp[idx]['Y'],temp[idx]['S'],'o',label = 'Full wave at {}m'.format(x))
        plt.plot(temp[idx]['Y'],temp[idx]['ICNIRP Average'],'+' ,label ='ICNIRP at {}m'.format(x))
        plt.plot(temp[idx]['Y'],temp[idx]['OET65'],'-' ,label ='OET 65 at {}m'.format(x))
    plt.legend()
    plt.xlabel('Y (m)')
    plt.ylabel('S (W/m)')
    plt.ylim([0,20])
    plt.title( 'Comparing various simulation methods of a 900Mhz,80W sector antenna at various x positions')
    plt.show()

def plotByCylindrical(df):
    R  = np.linspace(1, 10, 5)
    for r in R:
        temp = df.loc[(df['R'] <= r) & (df['R'] > r-0.05)]
        idx = temp.groupby(['phi','theta'])['R'].transform(max) == temp['R']
        phi = np.round(temp[idx]['phi'],2)
        theta = np.round(temp[idx]['theta'],2)
        x = temp[idx]['R']*np.sin(theta)*np.cos(phi)
        y = temp[idx]['R']*np.sin(theta)*np.sin(phi)
        z = temp[idx]['R']*np.cos(theta)
        plt.plot(temp[idx]['Y'],temp[idx]['S'],label = r)
        plt.legend()
    plt.show()


# all Sector coverage arrays
def AverageCylindricalSector(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0,ry = None):
    G = 10**(G/10)
    AHPBW = np.pi*AHPBW/180
    ro = AHPBW*G*L*np.cos(y)**2/12
    ry = R/np.cos(y)
    return P*2**(-1*(2*phi/AHPBW)**2)/(AHPBW*ry*L*(np.cos(y)**2)*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalSector(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0):
    G = 10**(G/10)
    AHPBW *= np.pi/180
    ro = AHPBW*G*L*np.cos(y)**2/12
    ry = R/np.cos(y)
    return 2*P*2**(-4*(phi/AHPBW)**2)/(AHPBW*ry*L*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def AdjustedSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    b2 = 1.9*phi/AHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2-b2**2)
    return 1.2*power*Gphitheta/(4*np.pi*R**2)

def SimpleSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = 0, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    b2 = (2*phi/AHPBW)**2
    Gphitheta = Globe + G*2**(-b1-b2)
    return power*Gphitheta/(4*np.pi*R**2)


def AverageCylindricalOmni(R, power = 80, VHPBW = 8.5, AHPBW = 85,G = 17, L =2.25, y = 0 ):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*L*np.cos(y)**2/2
    return  power/(2*np.pi*ry*L*np.cos(y)**2*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalOmni(R, power = 80, L = 2.25, G = 17, y = 0):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*L*np.cos(y)**2/2
    return  power/(np.pi*ry*L*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def SimpleSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    Gphitheta = Globe + G*2**(-b1)
    return power*Gphitheta/(4*np.pi*R**2)

def AdjustedSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2)
    return power*Gphitheta/(4*np.pi*R**2)

def Classical(E):
    S = []
    for i in range(len(E)):
        S.append(E[i]**2/377)
    return np.array(S)

def getGain(df):
    dfG = GetFarField('IEC-62232-panel-antenna (4)_FarField1.ffe')
    df['phi'] = np.abs(np.round(df['phi']*180/np.pi))
    df['theta'] = np.abs(np.round(df['theta']*180/np.pi))
    df = df.merge(dfG,how='left',on=['phi','theta'])
    return df['Gain']
def Validationtest1():

    FCCoccupational = 30#getZone(900,'FCC')[1]
    l1 = [65.8, 39.4, 25, 13.4, 10.7, 10.4, 10.1, 9.44]
    l2 = [6.22, 9.16, 13.6, 19.6, 26.3, 32.8, 37.6, 39.4, 37.6, 32.8, 26.3, 19.6, 13.6, 9.16, 6.22]
    l3 = [0.173, 3.77, 129, 105, 111, 163, 272, 338, 272, 163, 111, 105, 129, 3.77, 0.173]

    IXUS1_persentage_occupation = [243.8, 138.7, 84.43, 43.62, 36.72, 36.46, 35.51, 33.07]
    IXUS2_persentage_occupation = [19.2, 29.18, 44.46, 66.19, 91.01, 114.7, 133, 138.7, 132.1, 114.1, 90.44, 65.52, 44.46, 29.18, 19.19]
    IXUS3_persentage_occupation = [3.623, 24.51, 595, 273.1, 508.7, 682.9, 869.8, 1279, 886.2, 683.6, 505.8, 267.4, 594.2, 24.24, 3.706]

    def Doline(*lines):
        i = 0
        for line in lines:
            figure, axis = plt.subplots(2)
            axis[0].plot(line['1D'], line['l'],'k-',label = 'IEC Full wave refernce results')
            axis[0].plot(line['1D'], line['IXUS'],'k--',label = 'IXUS')
            axis[0].plot(line['1D'], line['Classical'],'k:', label = 'S=|E|^2/377')
            axis[0].plot(line['1D'], line['Full wave'],'k-.', label = 'S=ExH')
            axis[0].plot(line['1D'], line['OET65'],'k_', label = 'FCC OET 65')
            axis[0].plot(line['1D'], line['IEC Peak'],'ko', label = 'IEC Peak Estimation')
            axis[0].plot(line['1D'], line['EMSS Peak'],'k*', label = 'EMSS Peak Estimation')
            line = line.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|'])
            #plt.plot(line['1D'], line['IEC Average'], label = 'IEC Average')
            axis[1].plot(line['1D'],100 - line['IXUS']/line['l']*100,'k--',label = 'IXUS')
            axis[1].plot(line['1D'],100 - line['Classical']/line['l']*100,'k:', label = 'S=|E|^2/377')
            axis[1].plot(line['1D'],100 - line['Full wave']/line['l']*100,'k-.', label = 'S=ExH')
            axis[1].plot(line['1D'],100 - line['OET65']/line['l']*100,'k_', label = 'FCC OET 65')
            axis[1].plot(line['1D'],100 - line['IEC Peak']/line['l']*100,'ko', label = 'IEC Peak Estimation')
            axis[1].plot(line['1D'],100 - line['EMSS Peak']/line['l']*100,'k*', label = 'EMSS Peak Estimation')

            if i ==0:
                axis[0].set_title('Validation test for line 1')
                axis[0].set_xlabel('x(m)') 
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 1')
                axis[1].set_xlabel('x(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
                #print(np.corrcoef(line['l'], line['Full wave'])[0,1])
                #print(np.corrcoef(line['l'], line['Classical'])[0,1])            
                #print(np.corrcoef(line['l'], line['IXUS'])[0,1])            
                #print(np.corrcoef(line['l'], line['OET65'])[0,1])            
                #print(np.corrcoef(line['l'], line['IEC Peak'])[0,1])   
                #print(np.corrcoef(line['l'], line['EMSS Peak'])[0,1])  

                print(np.max(np.abs(line['l'] - line['Full wave'])))
                print(np.max(np.abs(line['l'] - line['Classical'])))
                print(np.max(np.abs(line['l'] - line['IXUS'])))
                print(np.max(np.abs(line['l'] - line['OET65'])))
                print(np.max(np.abs(line['l'] - line['IEC Peak'])))
                print(np.max(np.abs(line['l'] - line['EMSS Peak'])))

            if i ==1:
                axis[0].set_title('Validation test for line 2')
                axis[0].set_xlabel('y(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 2')
                axis[1].set_xlabel('y(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
                
            if i ==2:
                axis[0].set_title('Validation test for line 3')
                axis[0].set_xlabel('z(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 3')
                axis[1].set_xlabel('z(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
            i+=1
            figure.legend(axis[0].get_legend_handles_labels()[0],axis[0].get_legend_handles_labels()[1])
            figure.tight_layout()
        plt.show()







    line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False, power=80).df
    line1['IXUS'] = [x/100*FCCoccupational for x in IXUS1_persentage_occupation]
    line1['l'] = l1

    line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False, power=80).df
    line2['IXUS'] = [x/100*FCCoccupational for x in IXUS2_persentage_occupation]
    line2['l'] = l2

    line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False, power=80).df
    line3['IXUS'] = [x/100*FCCoccupational for x in IXUS3_persentage_occupation]
    line3['l'] = l3

    Doline(line1.rename(columns = {'X': '1D'}), line2.rename(columns = {'Y': '1D'}),line3.rename(columns = {'Z': '1D'}))
    


def CylindricalValidationTest():
    SectorSpacialAverage = [5.58, 3.54, 2.49, 1.86, 1.43, 1.02, 0.639]
    SectorSpacialPeak = [9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727]
    IXUSAverage_percentage = [103.4, 49.87, 28.77, 18.61, 13, 8.368, 4.727]
    IXUSAverage= [p/100*6 for p in IXUSAverage_percentage]



    FEKO1 = GetField('IEC-62232-panel-antenna (5)_NearField1.efe','IEC-62232-panel-antenna (5)_NearField1.hfe')
    FEKO2 = GetField('IEC-62232-panel-antenna (5)_NearField2.efe','IEC-62232-panel-antenna (5)_NearField2.hfe')
    FEKO3 = GetField('IEC-62232-panel-antenna (5)_NearField3.efe','IEC-62232-panel-antenna (5)_NearField3.hfe')
    FEKO4 = GetField('IEC-62232-panel-antenna (5)_NearField4.efe','IEC-62232-panel-antenna (5)_NearField4.hfe')
    FEKO5 = GetField('IEC-62232-panel-antenna (5)_NearField5.efe','IEC-62232-panel-antenna (5)_NearField5.hfe')
    FEKO6 = GetField('IEC-62232-panel-antenna (5)_NearField6.efe','IEC-62232-panel-antenna (5)_NearField6.hfe')
    FEKO7 = GetField('IEC-62232-panel-antenna (5)_NearField7.efe','IEC-62232-panel-antenna (5)_NearField7.hfe')
    Fullwave1 = FEKO1.df['Full wave'].to_numpy()[0]
    Fullwave2 = FEKO2.df['Full wave'].to_numpy()[0]
    Fullwave3 = FEKO3.df['Full wave'].to_numpy()[0]
    Fullwave4 = FEKO4.df['Full wave'].to_numpy()[0]
    Fullwave5 = FEKO5.df['Full wave'].to_numpy()[0]
    Fullwave6 = FEKO6.df['Full wave'].to_numpy()[0]
    Fullwave7 = FEKO7.df['Full wave'].to_numpy()[0]
    classical1 = FEKO1.df['Classical'].to_numpy()[0]
    classical2 = FEKO2.df['Classical'].to_numpy()[0]
    classical3 = FEKO3.df['Classical'].to_numpy()[0]
    classical4 = FEKO4.df['Classical'].to_numpy()[0]
    classical5 = FEKO5.df['Classical'].to_numpy()[0]
    classical6 = FEKO6.df['Classical'].to_numpy()[0]
    classical7 = FEKO7.df['Classical'].to_numpy()[0]


    Fullwave = np.array([Fullwave1, Fullwave2, Fullwave3, Fullwave4,Fullwave5, Fullwave6, Fullwave7])
    classical = np.array([classical1, classical2, classical3, classical4, classical5, classical6, classical7])
    print(Fullwave)
    print(classical)
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    L = 2.158
    AHPBW = 84
    y = 5
    Gs = 17 #dBi
    Go = 11 #dBi
    Globe = -9      #dBi
    Globe = -3.6    #dBi
    phi = np.pi/12
    theta = np.pi
    Ry = [4, 6, 8, 10, 12, 15, 20]
    R = [ry*np.cos(y*np.pi/180) for ry in Ry]

    IECSectorAverage= AverageCylindricalSector(phi,R,power,AHPBW, L,Gs, y*np.pi/180,Ry)
    IECSectorPeak = PeakCylindricalSector(phi,R,power,AHPBW, L, Gs, y=y*np.pi/180)
    df = pd.DataFrame(Ry, columns=['R'])
    df['phi'] = np.array([phi for r in R])
    df['theta'] = np.array([theta for r in R])
    OET65 = OET65mesh(R,getGain(df),D = L)

    plt.figure()
    plt.plot(Ry,IECSectorAverage,'kD',label = 'IEC Average Estimation')
    plt.plot(Ry,IECSectorAverage,'k*',label = 'EMSS Average Estimation')
    plt.plot(Ry,SectorSpacialAverage,'k:',label = 'Sector-coverage Spacial-average reference results')
    plt.plot(Ry,OET65,'k-.',label = 'FCC OET 65')
    plt.plot(Ry,IXUSAverage,'k_',label = 'IXUS')
    plt.plot(Ry,Fullwave/2,'k--',label = 'S = ExH')
    plt.plot(Ry,classical/2,'ko',label = 'S=|E|^2/377')
    plt.ylabel('S(W/m^2)')
    plt.xlabel('Ry (m)')
    plt.title('Validation test 2 Sector-coverage average results')
    plt.legend()

    plt.figure()
    plt.plot(Ry,IECSectorPeak,'kD',label = 'IEC Peak Estimation')
    plt.plot(Ry,IECSectorPeak,'k*',label = 'EMSS Peak Estimation')
    plt.plot(Ry,SectorSpacialPeak,'k:',label = 'Sector-coverage Spacial-peak reference results')
    plt.plot(Ry,OET65,'k-.',label = 'FCC OET 65')
    plt.plot(Ry,Fullwave,'k--',label = 'S = ExH')
    plt.plot(Ry,classical,'ko',label = 'S=|E|^2/377')
    plt.ylabel('S(W/m^2)')
    plt.xlabel('Ry (m)')
    plt.title('Validation test 2 Sector-coverage peak results')
    plt.legend()
    plt.show()


def SphericalValidationTest():
    adjustedSectorS = [52, 353, 313, 210, 141, 98.6, 72, 54.5 ]
    adjustedSectorS = [a/1000 for a in adjustedSectorS]
    SectorCoverageS = [9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727]
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    L = 2.158
    Gs = 17 #dBi
    Go = 11 #dBi
    Globes = -3.6     #dBi
    Globe0 = -9   #dBi
    phi = np.pi/12
    Ry = np.linspace(10,80,8)
    R = [np.sqrt(ry**2 + 5**2) for ry in Ry]
    R = np.array(R)
    theta = [np.pi/2 + np.arctan(5/ry) for ry in Ry]
    theta = np.array(theta)
    Ry = np.array(Ry)

    adjustedSpherical = AdjustedSphericalSector(theta = theta, phi=phi, R = Ry,power = power, VHPBW=8, AHPBW=84, L = L, G=Gs, Globe=Globes, y =5*np.pi/180 )
    plt.figure()
    plt.plot(Ry,adjustedSpherical,label = 'SpacialPeakCylindrical')
    plt.plot(Ry,adjustedSectorS,label = 'Adjusted Spherical Validation line')
    #plt.plot(Ry,SectorCoverageSbar,label = 'Average Cylindrical Validation line')
    #plt.plot(Ry,SectorPeak,label = 'SpacialAverageCylindrical')
    plt.legend()
    plt.show()



##Near field
def getEfficiency(G = 17, f = 900,A = 2.25*0.3):
    lamda = 3*10**8/(f*10**6)
    #return (10**(G/10)*lamda**2)/(4*np.pi*A)

    top = 10**(G/10)*lamda**2/(4*np.pi)
    bottom =  np.pi*2.25**2/4#np.pi*2.25**2/4
    return top/bottom

def Ssurface(P = 80, A = 2.25*0.3):
    return 4*P/A

def Snf(G = 17, f = 900,w = 0.3,D = 2.25,power = 80):
    A = w*D
    n = getEfficiency(G, f, A)
    return 16*n*power/(np.pi*D**2)

def St(R):
    return Snf()*Rnf()/R


def Rnf(D = 2.25, f = 900):
    lamda = 3*10**8/(f*10**6)
    return D**2/(4*lamda)

def Rff(D = 2.25,f = 900):
    lamda = 3*10**8/(f*10**6)
    return 0.6*D**2/lamda

def Sff(R, power = 80, G = 17):
    return power*10**(G/10)/(4*np.pi*R**2) 

def OET65near(R, power = 80, D = 2.25, AHPBW = 85):
    AHPBW *=np.pi/180
    return power/(R*D*AHPBW)


def OET65far(R,G,power = 80):
    G = 10**(G/10)
    return power*G/(4*np.pi*R**2)

def OET65Modified(D = 2.25):
    Rtrans = D*1.5
    Rfar = Rff()
    return OET65near(Rtrans)*1/(Rfar/Rtrans)**2

def OET65mesh(R, G, D = 2.25, f = 900):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S =[]
    for i in range(len(R)):
        if R[i] < Rnearfield:
            S.append(OET65near(R[i]))
        elif R[i] > Rnearfield:
            S.append(OET65far(R[i],G[i]))
    return np.array(S)
    

def OET65mesh2(R, f = 900,D = 2.25, a = True):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S = []
    if (a == True):
        for i in range(len(R)):
            if R[i] < 0.5:
                S.append(Ssurface())
            elif np.abs(R[i]) < Rnf():
                S.append(Snf())
            elif np.abs(R[i]) > Rff():
                S.append(Sff(R[i]))
            else:
                S.append(St(R[i]))
        return np.array(S)
    else:
        for i in range(len(R)):
            if np.abs(R[i]) < Rnearfield:
                S.append(Snf())
            elif np.abs(R[i]) > Rnearfield:
                S.append(Sff(R[i]))
            else:
                S.append(St(R[i]))
        return np.array(S)

def IECSpatialPeakSectorBasic(R, power = 80, D = 2.25, AHPBW = 85):
    AHPBW *=np.pi/180
    return 2*power/(R*D*AHPBW)

def IECSpatialPeakOmniBasic(R, power = 80, D = 2.25):
    AHPBW *=np.pi/180
    return power/(R*D*np.pi)

def IECSpatialAverageSectorBasic(R, power = 80, D = 2.25, AHPBW = 85):
    AHPBW *=np.pi/180
    return power/(R*D*AHPBW)

def IECSpatialAverageOmniBasic(R, power = 80, D = 2.25):
    AHPBW *=np.pi/180
    return power/(R*D*2*np.pi)


def IECmeshPeakSector(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if np.abs(phi[i]) < np.pi/2 and np.abs(R[i]*np.cos(theta[i])) < D/2 :
                S.append(PeakCylindricalSector(phi[i],R[i]))
            else:
                S.append(AdjustedSphericalSector(theta[i], phi[i], R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(IECSpatialPeakSectorBasic(R))  
    return np.array(S)

def IECmeshAverageSector(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if np.abs(phi[i]) < np.pi/2:
                S.append(AverageCylindricalSector(phi[i],R[i]))
            else:
                S.append(AdjustedSphericalSector(theta[i], phi[i], R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(IECSpatialAverageSectorBasic(R))  
    return np.array(S)

def EMSSmeshPeakSector(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if np.abs(phi[i]) < np.pi/2:
                S.append(PeakCylindricalSector(phi[i],R[i]))
            else:
                S.append(AdjustedSphericalSector(theta[i], phi[i], R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(SimpleSphericalSector(theta[i], phi[i], R[i]))   
    return np.array(S)

def EMSSmeshAverageSector(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            S.append(AverageCylindricalSector(phi[i],R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(SimpleSphericalSector(theta[i], phi[i], R[i]))
    return np.array(S)