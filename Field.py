
from cgitb import small
from matplotlib import pyplot as plt
import matplotlib

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
            print('here')
            maxFreq = getZone(self.f,standard)[1]
            minFreq = getZone(self.f,standard)[0]  
            self.df['Restriction'] = 1
            self.df.loc[minFreq > self.df['S'],'Restriction'] = 0
            self.df.loc[maxFreq < self.df['S'],'Restriction'] = 2

class Fekofield(Field):
    def __init__(self,power,source,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df):
        super().__init__(df,frequency*(10**-6),type = 'Feko',standard=standard, restriction = True)
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

    def plot2DZones(self,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True,S = 'Full wave'):
        colors = {0:Ncolor,1:GPcolor,2:Ocolor}
        labels = ['Safe','General Public','Occupational']
        #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))

        groups = self.df.groupby('Restriction')
       
        fig, ax = plt.subplots(1, figsize=(xfig,yfig))
        for label, group in groups:
            ax.scatter(group[axis1], group[axis2], 
                    c=group['Restriction'].map(colors), label=label)

        ax.set(xlabel= axis1, ylabel=axis2)
        ax.set_title("Restricions with {} simulation method".format(S))
        ax.legend(labels =labels, title='Restriction levels')
        if show:
            plt.show()

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



      
def GetField(filenameE,filenameH,S = 'Full wave',compress = False ,standard = 'FCC',power = 80):
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

    df['Full wave'] = np.sqrt(np.absolute(df['Sx'])**2 + np.absolute(df['Sy'])**2 + np.absolute(df['Sz'])**2)/2
    df['Classical'] = Classical(df['|E|'].to_numpy())/2
    df['OET65'] = OET65mesh(df['R'],df['Gain'])
#def IECmeshPeakSector(R, phi, theta,power = 80, f = 900, D = 2.25,y = 5*np.pi/180,G = 17, Globe = 11, VHPBW=8, AHPBW=84):

    df['IEC Peak'] = IECmeshPeakSector(df['R'], df['phi'], df['theta'],df['Gain'],power = 80, f = 900, D = 2.25,y =0,G = 17, Globe = -3.6, VHPBW=8.5, AHPBW=85)
    df['IEC Average'] = IECmeshAverageSector(df['R'], df['phi'], df['theta'],power = 80, f = 900, D = 2.25,y =0,G = 17, Globe = 11, VHPBW=8.5, AHPBW=85)
    df['EMSS Peak'] = EMSSmeshPeakSector(df['R'], df['phi'], df['theta'],power = 80, f = 900, D = 2.25,y =0,G = 17, Globe = 11, VHPBW=8.5, AHPBW=85)
    df['EMSS Average'] = EMSSmeshAverageSector(df['R'], df['phi'], df['theta'],power = 80, f = 900, D = 2.25,y =0,G = 17, Globe = 11, VHPBW=8.5, AHPBW=85)


    df['S'] = 2*df[S]
    if compress:
        df = df.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|','Full wave'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)

   




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
def plotBySZones(df, *S,Y = 'y', X = 'x', error = 0.01,round = 3):
    legends = ['k_', 'k*', 'kD', 'k+', 'k--']
    axis = [0,0]
    figure, (axis[0],axis[1]) = plt.subplots(1,2)
    for s in S:
        methods = ['Full wave', 'IEC Peak', 'EMSS Peak', 'Classical', 'OET65']
        for method,legend in zip(methods,legends):
            temp = df.loc[(df[method] >= s-error) & (df[method] < s+error)]
            axis[0].plot(temp['X'],temp['Y'],legend ,label = '{}'.format(method))
        axis[0].set_xlabel('X (m)')
        axis[0].set_ylabel('Y (m)')
        axis[0].set_title('Peak estimations')
        
        methods = ['Full wave', 'IEC Average', 'EMSS Average','Classical', 'OET65']
        df['Full wave'] = df['Full wave']/2
        df['Classical'] = df['Classical']/2
        for method,legend in zip(methods,legends):
            temp = df.loc[(df[method] >= s-error) & (df[method] < s+error)]
            axis[1].plot(temp['X'],temp['Y'], legend,label = '{} '.format(method))
        axis[1].set_xlabel('X (m)')
        axis[1].set_ylabel('Y (m)')
        axis[1].set_title('Average estimations')
    figure.suptitle('Comparing various simulation methods of a 900Mhz,80W sector antenna at S = 6W/m^2')
    figure.legend(axis[0].get_legend_handles_labels()[0],axis[0].get_legend_handles_labels()[1],loc='upper right')
    figure.tight_layout()
    plt.show()


def plotByCartesian(df, *X,error = 0.1,mode='Peak'):
    legends = ['k:', 'k--', 'k--', 'k:', 'k-.']
    axis = [0,0]
    figure, (axis[0],axis[1]) = plt.subplots(1,2)
    methodsPeaks = ['Full wave', 'IEC Peak', 'EMSS Peak', 'Classical', 'OET65']
    methodsAverages = ['Full wave', 'IEC Average', 'EMSS Average', 'Classical', 'OET65']
    methodsLabels = ['Full wave', 'IEC', 'EMSS', 'Classical', 'OET65']

    for x in X:
        for methodsPeak,methodsAverage,methodslabel,legend in zip(methodsPeaks,methodsAverages,methodsLabels,legends):
            temp = df.loc[(df['X'] >= x-error) & (df['X'] < x+error)]
            idx = temp.groupby(['Z','Y'])['X'].transform(max) == temp['X']
            axis[0].plot(temp[idx]['Y'],temp[idx][methodsPeak],legend,label = '{}'.format(methodslabel))

            if (methodsAverage == 'Full wave') or (methodsAverage == 'Classical'):
                axis[1].plot(temp[idx]['Y'],temp[idx][methodsAverage]/2,legend ,label ='{}'.format(methodslabel))
            else:
                axis[1].plot(temp[idx]['Y'],temp[idx][methodsAverage],legend ,label ='{}'.format(methodslabel))



    axis[0].set_title('Peak estimations')
    axis[0].set_xlabel('X (m)')
    axis[0].set_ylabel('S (W/m^2)')
    axis[1].set_title('Average estimations')
    axis[1].set_xlabel('X (m)')
    axis[1].set_ylabel('S (W/m^2)')
    figure.suptitle('Comparing various simulation methods of a 900Mhz,80W sector antenna at various x positions')
    figure.legend(axis[0].get_legend_handles_labels()[0],axis[0].get_legend_handles_labels()[1],loc='upper right')
    figure.tight_layout()
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
def AverageCylindricalSector(phi,R,power = 80,AHPBW = 85,D = 2.25,G= 17,y = 0,ry = None):
    G = 10**(G/10)
    AHPBW = np.pi*AHPBW/180
    ro = AHPBW*D*G*np.cos(y)**2/12
    if ry is None:
        ry = R/np.cos(y)
    return power*2**(-2*(phi/AHPBW)**2)/(AHPBW*ry*D*(np.cos(y)**2)*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalSector(phi,R,power = 80,AHPBW = 85,D = 2.25,G= 17,y = 0):
    G = 10**(G/10)
    AHPBW *= np.pi/180
    ro = (AHPBW*D*G*np.cos(y)**2)/12
    ry = R/np.cos(y)
    return (2*power*2**(-2*(phi/AHPBW)**2))/(AHPBW*ry*D*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def AdjustedSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, D = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    b2 = 1.9*phi/AHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2-b2**2)
    return 1.2*power*Gphitheta/(4*np.pi*R**2)

def SimpleSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, D = 2.25, G = 17,Globe = 0, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    b2 = (2*phi/AHPBW)**2
    Gphitheta = Globe + G*2**(-b1-b2)
    return power*Gphitheta/(4*np.pi*R**2)


def AverageCylindricalOmni(R, power = 80, VHPBW = 8.5, AHPBW = 85,G = 17, D =2.25, y = 0 ):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*D*np.cos(y)**2/2
    return  power/(2*np.pi*ry*D*np.cos(y)**2*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalOmni(R, power = 80, D = 2.25, G = 17, y = 0):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*D*np.cos(y)**2/2
    return  power/(np.pi*ry*D*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def SimpleSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, D = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    Gphitheta = Globe + G*2**(-b1)
    return power*Gphitheta/(4*np.pi*R**2)

def AdjustedSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, D = 2.25, G = 17,Globe = -3.6, y = 0):
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
            axis = [0,0]
            figure, (axis[0],axis[1]) = plt.subplots(1,2)
            axis[0].plot(line['1D'], line['l'],'k-',label = 'IEC Full wave refernce results')
            axis[0].plot(line['1D'], line['IXUS'],'k--',label = 'IXUS')
            axis[0].plot(line['1D'], line['Classical'],'k:', label = 'S=|E|^2/377')
            axis[0].plot(line['1D'], line['Full wave'],'k-.', label = 'S=ExH')
            axis[0].plot(line['1D'], line['OET65'],'k_', label = 'FCC OET 65')
            axis[0].plot(line['1D'], line['IEC Peak'],'ko', label = 'IEC Peak Estimation')
            axis[0].plot(line['1D'], line['EMSS Peak'],'k*', label = 'EMSS Peak Estimation')
            line = line.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|'])
            #plt.plot(line['1D'], line['IEC Average'], label = 'IEC Average')
            axis[1].plot(line['1D'], (line['IXUS']-line['l'])/line['l']*100,'k--',label = 'IXUS')
            axis[1].plot(line['1D'], (line['Classical']-line['l'])/line['l']*100,'k:', label = 'S=|E|^2/377')
            axis[1].plot(line['1D'], (line['Full wave']-line['l'])/line['l']*100,'k-.', label = 'S=ExH')
            axis[1].plot(line['1D'], (line['OET65']-line['l'])/line['l']*100,'k_', label = 'FCC OET 65')
            axis[1].plot(line['1D'], (line['IEC Peak']-line['l'])/line['l']*100,'ko', label = 'IEC Peak Estimation')
            axis[1].plot(line['1D'], (line['EMSS Peak']-line['l'])/line['l']*100,'k*', label = 'EMSS Peak Estimation')
            axis[1].set_ylim([-100,100])
            if i ==0:
                axis[0].set_title('Validation test for line 1')
                axis[0].set_xlabel('x(m)') 
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage difference to reference results for line 1')
                axis[1].set_xlabel('x(m)')
                axis[1].set_ylabel('Percentage difference')
                #print(np.corrcoef(line['l'], line['Full wave'])[0,1])
                #print(np.corrcoef(line['l'], line['Classical'])[0,1])            
                #print(np.corrcoef(line['l'], line['IXUS'])[0,1])            
                #print(np.corrcoef(line['l'], line['OET65'])[0,1])            
                #print(np.corrcoef(line['l'], line['IEC Peak'])[0,1])   
                #print(np.corrcoef(line['l'], line['EMSS Peak'])[0,1])  

                #print(np.max(np.abs(line['l'] - line['Full wave'])))
                #print(np.max(np.abs(line['l'] - line['Classical'])))
                #print(np.max(np.abs(line['l'] - line['IXUS'])))
                #print(np.max(np.abs(line['l'] - line['OET65'])))
                #print(np.max(np.abs(line['l'] - line['IEC Peak'])))
                #print(np.max(np.abs(line['l'] - line['EMSS Peak'])))

            if i ==1:
                axis[0].set_title('Validation test for line 2')
                axis[0].set_xlabel('y(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage difference to reference results for line 2')
                axis[1].set_xlabel('y(m)')
                axis[1].set_ylabel('Percentage difference')
                
            if i ==2:
                axis[0].set_title('Validation test for line 3')
                axis[0].set_xlabel('z(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage difference to reference results for line 3')
                axis[1].set_xlabel('z(m)')
                axis[1].set_ylabel('Percentage difference')
            i+=1
            figure.legend(axis[0].get_legend_handles_labels()[0],axis[0].get_legend_handles_labels()[1],loc='upper center')
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
    SectorSpacialAverage = np.array([5.58, 3.54, 2.49, 1.86, 1.43, 1.02, 0.639])
    SectorSpacialPeak = np.array([9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727])
    IXUSAverage_percentage = [103.4, 49.87, 28.77, 18.61, 13, 8.368, 4.727]
    IXUSAverage= np.array([p/100*6 for p in IXUSAverage_percentage])

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
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    D = 2.158
    AHPBW = 84
    y = 5
    Gs = 17 #dBi
    Go = 11 #dBi
    Globe = -9      #dBi
    Globe = -3.6    #dBi
    phi = np.pi/12
    theta = np.pi
    Ry = np.array([4, 6, 8, 10, 12, 15, 20])
    R = [ry*np.cos(y*np.pi/180) for ry in Ry]

    IECSectorAverage= AverageCylindricalSector(phi,R,power,AHPBW, D,Gs, y*np.pi/180,Ry)
    IECSectorPeak = PeakCylindricalSector(phi,R,power,AHPBW, D, Gs, y=y*np.pi/180)
    df = pd.DataFrame(Ry, columns=['R'])
    df['phi'] = np.array([phi for r in R])
    df['theta'] = np.array([theta for r in R])
    OET65 = OET65mesh(Ry,getGain(df),D=D)


    legends = ['kD', 'k_', 'k:', 'k+', 'k--','k-.','ko']
    methods = ['IEC Estimation', 'EMSS Estimation','FCC OET 65','Ray Tracing','S = ExH' ,'S=|E|^2/377','Reference results']
    axis = [0,0]
    figure, (axis[0],axis[1]) = plt.subplots(1,2)
    axis[0].plot(Ry,IECSectorAverage,legends[0],label = 'IEC Average Estimation')
    axis[0].plot(Ry,IECSectorAverage,legends[1],label = 'EMSS Average Estimation')
    axis[0].plot(Ry,OET65,legends[2],label = 'FCC OET 65')
    axis[0].plot(Ry,IXUSAverage,legends[3],label = 'Ray Tracing')
    axis[0].plot(Ry,Fullwave,legends[4],label = 'S = ExH')
    axis[0].plot(Ry,classical,legends[5],label = 'S=|E|^2/377')
    axis[0].plot(Ry,SectorSpacialAverage,legends[6],label = 'Sector-coverage Spacial-average reference results')

    axis[0].set_ylabel('S(W/m^2)')
    axis[0].set_xlabel('Ry (m)')
    axis[0].set_title('Sector-coverage average results')
    axis[1].plot(Ry,(IECSectorAverage-SectorSpacialAverage)/SectorSpacialAverage*100,legends[0],label = 'IEC Average Estimation')
    axis[1].plot(Ry,(IECSectorAverage-SectorSpacialAverage)/SectorSpacialAverage*100,legends[1],label = 'EMSS Average Estimation')
    axis[1].plot(Ry,(OET65-SectorSpacialAverage)/SectorSpacialAverage*100,legends[2],label = 'FCC OET 65')
    axis[1].plot(Ry,(IXUSAverage-SectorSpacialAverage)/SectorSpacialAverage*100,legends[3],label = 'Ray Tracing')
    axis[1].plot(Ry,(Fullwave-SectorSpacialAverage)/SectorSpacialAverage*100,legends[4],label = 'S = ExH')
    axis[1].plot(Ry,(classical-SectorSpacialAverage)/SectorSpacialAverage*100,legends[5],label = 'S=|E|^2/377')
    axis[1].set_ylabel('Percentage of reference results')
    axis[1].set_xlabel('Ry (m)')
    axis[1].set_title('Percentage of  Peak results')
    figure.legend(labels=methods,loc='upper center')
    figure.tight_layout()

    figure, (axis[0],axis[1]) = plt.subplots(1,2)
    axis[0].plot(Ry,IECSectorPeak,legends[0],label = 'IEC Peak Estimation')
    axis[0].plot(Ry,IECSectorPeak,legends[1],label = 'EMSS Peak Estimation')
    axis[0].plot(Ry,OET65,legends[2],label = 'FCC OET 65')
    axis[0].plot(Ry,IXUSAverage,legends[3],label = 'Ray Tracing')
    axis[0].plot(Ry,Fullwave,legends[4],label = 'S = ExH')
    axis[0].plot(Ry,classical,legends[5],label = 'S=|E|^2/377')
    axis[0].plot(Ry,SectorSpacialPeak,legends[6],label = 'Sector-coverage Spacial-peak reference results')
    axis[0].set_ylabel('S(W/m^2)')
    axis[0].set_xlabel('Ry (m)')
    axis[0].set_title('Sector-coverage peak results')
    axis[1].plot(Ry,(IECSectorPeak-SectorSpacialPeak)/SectorSpacialPeak*100,legends[0],label = 'IEC Average Estimation')
    axis[1].plot(Ry,(IECSectorPeak-SectorSpacialPeak)/SectorSpacialPeak*100,legends[1],label = 'EMSS Average Estimation')
    axis[1].plot(Ry,(OET65-SectorSpacialPeak)/SectorSpacialPeak*100,legends[2],label = 'FCC OET 65')
    axis[1].plot(Ry,(IXUSAverage-SectorSpacialPeak)/SectorSpacialPeak*100,legends[3],label = 'Ray Tracing')
    axis[1].plot(Ry,(Fullwave-SectorSpacialPeak)/SectorSpacialPeak*100,legends[4],label = 'S = ExH')
    axis[1].plot(Ry,(classical-SectorSpacialPeak)/SectorSpacialPeak*100,legends[5],label = 'S=|E|^2/377')
    axis[1].set_ylabel('Percentage of reference results')
    axis[1].set_xlabel('Ry (m)')
    axis[1].set_title('Percentage of  Peak results')
    figure.legend(labels=methods,loc='upper center')
    figure.tight_layout()
    plt.show()


def SphericalValidationTest():
    adjustedSectorS = [52, 353, 313, 210, 141, 98.6, 72, 54.5 ]
    adjustedSectorS = [a/1000 for a in adjustedSectorS]
    simpleSectorS = [22.2, 26.5, 136, 150, 114, 81.1, 57.8, 42.1]
    simpleSectorS = np.array([a/1000 for a in simpleSectorS])
    IXUS_percentages = [0.2475, 0.03976, 0.03933, 0.02537, 0.004553, 0.004815, 0.01245, 0.02005]
    rayTracing = np.array([ixus/100*30 for ixus in IXUS_percentages])
    f = 925
    y = 5*np.pi/180
    lamda = (3*10**8)/(f*10**6)
    power = 80
    D = 2.158
    Gs = 17 #dBi
    Go = 11 #dBi
    Globe = -3.6     #dBi
    Globe0 = -9   #dBi
    d = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    R = np.array(np.sqrt(d**2 + 5**2))
    Ry = np.array(R/np.cos(y))
    theta = np.array(np.pi/2 + np.arctan(5/R))
    phi = np.array([np.pi/12 for i in R])
    df = pd.DataFrame(phi,columns = ['phi'])
    df['theta'] = theta

 
    IECSectorAverage = IECmeshAverageSector(theta = theta, phi=phi, R = R,power = power, VHPBW=8, AHPBW=84, D = D, G=Gs, Globe=Globe, y =y )
    EMSSSectorAverage = EMSSmeshAverageSector(theta = theta, phi=phi, R = R,power = power, VHPBW=8, AHPBW=84, D = D, G=Gs, Globe=Globe, y =y )
    IECSectorPeak = IECmeshPeakSector(theta = theta, phi=phi, R = R,power = power, VHPBW=8, AHPBW=84, D = D, G=Gs, Globe=Globe, y =y )
    EMSSSectorPeak = EMSSmeshPeakSector(theta = theta, phi=phi, R = R,power = power, VHPBW=8, AHPBW=84, D = D, G=Gs, Globe=Globe, y =y )
    OET65 = OET65mesh(R, getGain(df),D=D,theta = theta+y)


    FEKO1 = GetField('IEC-62232-panel-antenna (5)_spherical1.efe','IEC-62232-panel-antenna (5)_spherical1.hfe')
    FEKO2 = GetField('IEC-62232-panel-antenna (5)_spherical2.efe','IEC-62232-panel-antenna (5)_spherical2.hfe')
    FEKO3 = GetField('IEC-62232-panel-antenna (5)_spherical3.efe','IEC-62232-panel-antenna (5)_spherical3.hfe')
    FEKO4 = GetField('IEC-62232-panel-antenna (5)_spherical4.efe','IEC-62232-panel-antenna (5)_spherical4.hfe')
    FEKO5 = GetField('IEC-62232-panel-antenna (5)_spherical5.efe','IEC-62232-panel-antenna (5)_spherical5.hfe')
    FEKO6 = GetField('IEC-62232-panel-antenna (5)_spherical6.efe','IEC-62232-panel-antenna (5)_spherical6.hfe')
    FEKO7 = GetField('IEC-62232-panel-antenna (5)_spherical7.efe','IEC-62232-panel-antenna (5)_spherical7.hfe')
    FEKO8 = GetField('IEC-62232-panel-antenna (5)_spherical8.efe','IEC-62232-panel-antenna (5)_spherical8.hfe')


    Fullwave1 = FEKO1.df['Full wave'].to_numpy()[0]
    Fullwave2 = FEKO2.df['Full wave'].to_numpy()[0]
    Fullwave3 = FEKO3.df['Full wave'].to_numpy()[0]
    Fullwave4 = FEKO4.df['Full wave'].to_numpy()[0]
    Fullwave5 = FEKO5.df['Full wave'].to_numpy()[0]
    Fullwave6 = FEKO6.df['Full wave'].to_numpy()[0]
    Fullwave7 = FEKO7.df['Full wave'].to_numpy()[0]
    Fullwave8 = FEKO8.df['Full wave'].to_numpy()[0]

    classical1 = FEKO1.df['Classical'].to_numpy()[0]
    classical2 = FEKO2.df['Classical'].to_numpy()[0]
    classical3 = FEKO3.df['Classical'].to_numpy()[0]
    classical4 = FEKO4.df['Classical'].to_numpy()[0]
    classical5 = FEKO5.df['Classical'].to_numpy()[0]
    classical6 = FEKO6.df['Classical'].to_numpy()[0]
    classical7 = FEKO7.df['Classical'].to_numpy()[0]
    classical8 = FEKO8.df['Classical'].to_numpy()[0]

    Fullwave = np.array([Fullwave1, Fullwave2, Fullwave3, Fullwave4,Fullwave5, Fullwave6, Fullwave7,Fullwave8])
    classical = np.array([classical1, classical2, classical3, classical4, classical5, classical6, classical7,classical8])

    axis = [0,0]
    methods = ['S = ExH','S = |E|^2/377','Ray Tracing', 'IEC estimations', 'EMSS estimations', 'FCC OET 65','Reference Results']
    legends = ['k:', 'kD', 'k--', 'k+', 'k-.','ko','k-']
    figure, (axis[0],axis[1]) = plt.subplots(1,2)   
    axis[0].plot(d,Fullwave,legends[0],label='S = ExH')
    axis[0].plot(d,classical,legends[1],label='S = |E|^2/377')
    axis[0].plot(d,rayTracing,legends[2],label='Ray Tracing')
    axis[0].plot(d,IECSectorAverage,legends[3],label = 'IEC Sector-Coverage Average')
    axis[0].plot(d,EMSSSectorAverage,legends[4],label = 'EMSS Sector-Coverage Average')
    axis[0].plot(d,OET65,legends[5],label = 'FCC OET 65')
    axis[0].plot(d,adjustedSectorS,legends[6],label = 'Adjusted Spherical Validation line')
    axis[0].set_xlabel('Ry (m)')
    axis[0].set_ylabel('S (W/m^2)')
    axis[0].set_title('Spacial average results')

    axis[1].plot(d,(Fullwave-adjustedSectorS)/adjustedSectorS*100,legends[2],label = 'S = ExH')
    axis[1].plot(d,(classical-adjustedSectorS)/adjustedSectorS*100,legends[2],label = 'S = |E|^2/377')
    axis[1].plot(d,(rayTracing-adjustedSectorS)/adjustedSectorS*100,legends[2],label = 'Ray Tracing')
    axis[1].plot(d,(IECSectorAverage-adjustedSectorS)/adjustedSectorS*100,legends[3],label = 'IEC Sector-Coverage Average')
    axis[1].plot(d,(EMSSSectorAverage-adjustedSectorS)/adjustedSectorS*100,legends[4],label = 'EMSS Sector-Coverage Average')
    axis[1].plot(d,(OET65-adjustedSectorS)/adjustedSectorS*100,legends[5],label = 'FCC OET 65')
    axis[1].set_xlabel('Ry (m)')
    axis[1].set_ylabel('Percentage of reference results')
    axis[1].set_title('Percentage error')
    figure.suptitle('Validation test 3 Adjusted formula test')
    figure.legend(labels = methods,loc='center')
    figure.tight_layout()

    figure, (axis[0],axis[1]) = plt.subplots(1,2)   
    axis[0].plot(d,Fullwave,legends[0],label='S = ExH')
    axis[0].plot(d,classical,legends[1],label='S = |E|^2/377')
    axis[0].plot(d,rayTracing,legends[2],label='Ray Tracing')
    axis[0].plot(d,IECSectorAverage,legends[3],label = 'IEC Sector-Coverage Average')
    axis[0].plot(d,EMSSSectorAverage,legends[4],label = 'EMSS Sector-Coverage Average')
    axis[0].plot(d,OET65,legends[5],label = 'FCC OET 65')
    axis[0].plot(d,simpleSectorS,legends[6],label = 'Adjusted Spherical Validation line')
    axis[0].set_xlabel('Ry (m)')
    axis[0].set_ylabel('S (W/m^2)')
    axis[0].set_title('Spacial average results')

    axis[1].plot(d,(Fullwave-simpleSectorS)/simpleSectorS*100,legends[2],label = 'S = ExH')
    axis[1].plot(d,(classical-simpleSectorS)/simpleSectorS*100,legends[2],label = 'S = |E|^2/377')
    axis[1].plot(d,(rayTracing-simpleSectorS)/simpleSectorS*100,legends[2],label = 'Ray Tracing')
    axis[1].plot(d,(IECSectorAverage-simpleSectorS)/simpleSectorS*100,legends[3],label = 'IEC Sector-Coverage Average')
    axis[1].plot(d,(EMSSSectorAverage-simpleSectorS)/simpleSectorS*100,legends[4],label = 'EMSS Sector-Coverage Average')
    axis[1].plot(d,(OET65-simpleSectorS)/simpleSectorS*100,legends[5],label = 'FCC OET 65')
    axis[1].set_xlabel('Ry (m)')
    axis[1].set_ylabel('Percentage of reference results')
    axis[1].set_title('Percentage error')
    figure.suptitle('Validation test 3 simple formula test')
    figure.legend(labels = methods,loc='center')
    figure.tight_layout()

    plt.show()



##Near field
def getEfficiency(G = 42.8, f = 7550,A = 2.4**2*np.pi):
    lamda = 3*10**8/(f*10**6)
    return (10**(G/10)*lamda**2)/(4*np.pi*A)


def Ssurface(power = 80, D = 2.4):
    A = np.pi*D**2
    return 4*power/A

def Snf(G = 42.8, f = 7550,D = 2.4,power = 80):
    A = np.pi*D**2/4
    n = getEfficiency(G, f, A)
    return 16*1*power/(np.pi*D**2)

def St(R):
    return Snf()*Rnf()/R


def Rnf(D = 2.4, f = 7750):
    lamda = 3*10**8/(f*10**6)
    return D**2/(4*lamda)

def Rff(D = 2.4,f = 7750):
    lamda = 3*10**8/(f*10**6)
    return 0.6*D**2/lamda

def Sff(R, power = 80, G = 42.8):
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

def OET65mesh(R, G, D = 2.25, f = 900,theta = None):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S =[]
    for i in range(len(R)):
        if R[i] < Rnearfield:
            S.append(OET65near(R[i],D=D))
        elif R[i] > Rnearfield:
            S.append(OET65far(R[i],G[i]))
    return np.array(S)
    

def OET65meshApeture(R, f = 7550,D = 2.4):
    S = []
    for i in range(len(R)):
        if R[i] < 10:
            S.append(Ssurface(D=D/2))
        elif np.abs(R[i]) < Rnf(D = D):
            S.append(Snf())
        elif np.abs(R[i]) > Rff(D = D):
            S.append(Sff(R[i]))
        else:
            S.append(St(R[i]))
    print('near field ends at {}'.format(Rnf()))
    print('far field ends at {}'.format(Rff()))
    return np.array(S)


def IECSpatialPeakSectorBasic(R, power = 80, D = 2.25, AHPBW = 85):
    AHPBW *=2*np.pi/180
    return power/(R*D*AHPBW)

def IECSpatialPeakOmniBasic(R, power = 80, D = 2.25):
    AHPBW *=np.pi/180
    return power/(R*D*np.pi)

def IECSpatialAverageSectorBasic(R, power = 80, D = 2.25, AHPBW = 85):
    AHPBW *=np.pi/180
    return power/(2*R*D*AHPBW)

def IECSpatialAverageOmniBasic(R, power = 80, D = 2.25):
    AHPBW *=np.pi/180
    return power/(R*D*2*np.pi)


def IECmeshPeakSector(R, phi, theta,Gain,power = 80, f = 900, D = 2.25,y = 5*np.pi/180,G = 17, Globe = 11, VHPBW=8, AHPBW=84):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if np.abs(phi[i]) < np.pi/2 and np.abs(R[i]*np.cos(theta[i])) < D/2 :
                S.append(PeakCylindricalSector(phi[i],R[i],power = power, D = D,y = y,G = G, AHPBW=AHPBW))
            else:
                S.append(AdjustedSphericalSector(theta[i], phi[i], R[i],power = power, VHPBW = VHPBW,AHPBW = AHPBW, D = D, G = G,Globe = Globe, y = y))
        elif np.abs(R[i]) > Rnearfield:
            S.append(OET65far(R[i],Gain[i],power = power)) 
        
    return np.array(S)


def IECmeshAverageSector(R, phi, theta,power = 80, f = 900, D = 2.25,y = 5*np.pi/180,G = 17, Globe = 11, VHPBW=8, AHPBW=84):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if (np.abs(phi[i]) < np.pi/2) and (np.abs(R[i]*np.cos(theta[i])) < D/2):
                S.append(AverageCylindricalSector(phi=phi[i], R = R[i],power = power, AHPBW=AHPBW, D = D, G=G, y =y ))
            else:
                S.append(AdjustedSphericalSector(theta = theta[i], phi=phi[i], R = R[i],power = power, VHPBW=8, AHPBW=84, D = D, G=G, Globe=Globe, y =y ))
        elif np.abs(R[i]) > Rnearfield:
            S.append(IECSpatialAverageSectorBasic(power = power,R =R[i],D = D))  
    return np.array(S)

def EMSSmeshPeakSector(R, phi, theta,power = 80, f = 900, D = 2.25,y = 5*np.pi/180,G = 17, Globe = 11, VHPBW=8, AHPBW=84):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if np.abs(phi[i]) < np.pi/2:
                S.append(PeakCylindricalSector(phi[i],R[i]))
            else:
                S.append(AdjustedSphericalSector(phi[i],R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(SimpleSphericalSector(theta[i], phi[i], R[i],power = power, VHPBW=VHPBW, AHPBW=AHPBW, D=D, G=G, Globe=Globe,y=y )) 
    return np.array(S)

def EMSSmeshAverageSector(R, phi, theta,power = 80, f = 900, D = 2.25,y = 5*np.pi/180,G = 17, Globe = 11, VHPBW=8, AHPBW=84):
    lamda = 3*10**8/(f*10**6)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            if (np.abs(phi[i]) < np.pi/2) and (np.abs(R[i]*np.cos(theta[i])) < D/2):
                S.append(AverageCylindricalSector(phi=phi[i], R = R[i],power = power, AHPBW=AHPBW, D = D, G=G, y =y ))
            else:
                S.append(AdjustedSphericalSector(theta = theta[i], phi=phi[i], R = R[i],power = power, VHPBW=8, AHPBW=84, D = D, G=G, Globe=Globe, y =y ))
        elif np.abs(R[i]) > Rnearfield:
            S.append(SimpleSphericalSector(theta[i], phi[i], R[i],power = power, VHPBW=VHPBW, AHPBW=AHPBW, D=D, G=G, Globe=Globe,y=y )) 
    return np.array(S)

def AndrewAntennaTest():
    IXUS_percentages = np.array([
    1806,1806,1806,1806,1806,1806,1806,1806,1806,1806,
    1806,902.8,902.8,902.8,902.8,902.8,902.8,902.8,902.8,902.8,
    902.8,902.8,842.5,773.7,713.1, 659.3, 611.3, 568.4,
    529.9, 495.2, 463.7, 435.2, 409.2, 385.5, 363.8, 343.8, 325.5, 308.6,
    293,278.5,265.1,252.6,241,230.2,220.1,210.6,201.7,193.4,
    185.6,178.2,171.3,164.8,158.6,152.8,147.3,142.1,137.2,132.5,
    128,123.8
    ])

    print(len(IXUS_percentages))


    IXUS = IXUS_percentages/100*6
    R = np.array(np.linspace(0.5, 100, 60))
    S = OET65meshApeture(R)
    print(S)
    plt.plot(R,S,'k-.',label = 'FCC OET 65')
    plt.plot(R,IXUS,'k-', label = 'Ray Tracing')
    plt.title('Comparison between FCC OET 65 and IXUS simulations of a apeture antenna')
    plt.xlabel('X m')
    plt.ylabel('S W/m^2')

    plt.legend()

def validationtest4():
    x = np.array(np.linspace(30,100,8))
    y = 0
    z = -15
    referenceResults = np.array([2.33, 48.5, 105, 80.7, 45.2, 22.6, 10.7, 4.99])
    IXUS_percentage = np.array([0.04925, 1.079, 2.337, 1.794, 1.05, 0.498, 0.2353, 0.1109])
    rayTracing = IXUS_percentage/100*6

    xEff = x*np.cos(15*np.pi/180)
    zEff = z + x*np.sin(15*np.pi/180)
    Reff = np.sqrt((xEff)**2 + (zEff)**2)
    thetaEff = np.pi/2 + np.arcsin(zEff/Reff)
    phiEff = 0
    df = pd.DataFrame(xEff,columns = ['X'])
    df['Y'] = 0
    df['Z'] = zEff
    df['R'] = Reff
    df['theta'] = np.round(thetaEff*180/np.pi)
    df['phi'] = np.round(phiEff*180/np.pi)
    G = GetFarField('IEC-62232-panel-antenna (4)_FarField1.ffe')
    df = df.merge(G,how='left',on=['phi','theta'])
    df['theta'] = thetaEff
    df['phi'] = phiEff
    df['OET65']= OET65mesh(Reff,df['Gain'],f = 90)*1000
    print(df['Gain'])
    df['IEC'] = IECmeshPeakSector(df['R'],df['phi'],df['theta'],df['Gain'],power = 80, f=90,D=2.25,y = 0, G = 17, Globe = 0,VHPBW=8.5, AHPBW=85 )*1000
    df['EMSS'] = EMSSmeshPeakSector(df['R'],df['phi'],df['theta'],power = 80, f=90,D=2.25,y = 0, G = 17, Globe = 0,VHPBW=8.5, AHPBW=85 )*1000
    df['referenceResults'] = referenceResults
    df['Ray Tracing'] = rayTracing*1000
    Feko = GetField('IEC-62232-panel-antenna_validationTest4.efe','IEC-62232-panel-antenna_validationTest4.hfe').df
    df['S=ExH'] = Feko['Full wave']*1000
    df['S=|E|^2/377'] = Feko['Classical']*1000
    #df['']


    #plotting
    methods = ['S=ExH','S=|E|^2/377','FCC OET 65', 'IEC estimations', 'EMSS estimations','Ray Tracing' ,'Reference results']
    markers = ['kD','k:','k--','kD','k+','k-.','k_']
    columns = ['S=ExH','S=|E|^2/377','OET65', 'IEC', 'EMSS','Ray Tracing','referenceResults']
    axis = [0,0]
    figure, (axis[0],axis[1]) = plt.subplots(1,2) 
    for method,marker,column in zip(methods,markers,columns):
        axis[0].plot(df['X'],df[column],marker,label = method)
        axis[1].plot(df['X'],np.abs(100*(df[column]/1000-df['referenceResults']/1000)/df['referenceResults']/1000),marker,label = method)
    axis[0].set_xlabel('X (m)')
    axis[0].set_ylabel('S (mW/m^2)')
    axis[0].set_title('Far field results')

    axis[1].set_xlabel('X (m)')
    axis[1].set_ylabel('Percentage of reference results')
    axis[1].set_title('Percentage of far field results')

    figure.suptitle('Validation test 4 Results')
    figure.tight_layout()
    figure.legend(labels=methods,loc = 'center')
    plt.show()

    print(df)
