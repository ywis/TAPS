import pandas as pd
import numpy as np
import matplotlib.backends.backend_tkagg
import matplotlib.pylab as plt
from astropy.io import fits
from astropy import units as units
import astropy.io.fits as pyfits
from astropy.convolution import Gaussian1DKernel, convolve
from extinction import calzetti00, apply, ccm89
from scipy import optimize,interpolate
import sys
import time
import emcee
import corner
import math
from multiprocessing import Pool,cpu_count

import warnings
warnings.filterwarnings('ignore')

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

plt.tight_layout()
plt.rc('lines', linewidth=1, markersize=2)
plt.rc('font', size=12, family='serif')
plt.rc('mathtext', fontset='stix')
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=1.5, size=4)
plt.rc('ytick.major', width=1.5, size=4)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.subplots_adjust(bottom=0.2, left=0.2)

### Reading Spectra
df_cat=pd.read_csv('/home/siqi/goodss_3dhst_v4.1.5_catalogs/goodss_3dhst.v4.1.5.zbest.rf', delim_whitespace=True,header=None,comment='#',index_col=False)
df_cat.columns=["id", "z_best", "z_type", "z_spec", "DM", "L153", "nfilt153","L154","nfilt154", "L155", "nfilt155", "L161", "nfilt161", "L162", "nfilt162",\
                "L163", "nfilt163", "L156", "nfilt156", "L157", "nfilt157", "L158", "nfilt158", "L159", "nfilt159", "L160", "nfilt160", "L135", "nfilt135", "L136", "nfilt136",\
                "L137", "nfilt137", "L138", "nfilt138", "L139", "nfilt139", "L270", "nfilt270", "L271", "nfilt271", "L272", "nfilt272", "L273", "nfilt273", "L274", "nfilt274", "L275", "nfilt275"]


df = pd.read_csv('/home/siqi/TAPS/TAPS/source_list/matching_galaxies_goodss_20200317_PSB.csv', sep=',')
df.columns=['detector','ID','region','filename','chip']

df_photometry=pd.read_csv('/home/siqi/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', delim_whitespace=True,header=None,comment='#',index_col=False)
df_photometry.columns=["id", "x", "y", "ra", "dec", "faper_F160W", "eaper_F160W", "faper_F140W", "eaper_F140W", "f_F160W", "e_F160W", "w_F160W",\
                        "f_U38", "e_U38", "w_U38","f_U", "e_U", "w_U","f_F435W", "e_F435W", "w_F435W", "f_B", "e_B", "w_B", "f_V", "e_V", "w_V", \
                        "f_F606Wcand", "e_F606Wcand", "w_F606Wcand","f_F606W", "e_F606W","w_F606W","f_R", "e_R", "w_R", "f_Rc", "e_Rc", "w_Rc", \
                        "f_F775W", "e_F775W", "w_F775W","f_I", "e_I", "w_I", "f_F814Wcand", "e_F814Wcand", "w_F814Wcand", "f_F850LP", "e_F850LP", "w_F850LP",\
                        "f_F850LPcand", "e_F850LPcand", "w_F850LPcand", "f_F125W", "e_F125W", "w_F125W","f_J", "e_J", "w_J", "f_tenisJ", "e_tenisJ", "w_tenisJ",\
                        "f_F140W", "e_F140W", "w_F140W","f_H", "e_H", "w_H", "f_tenisK", "e_tenisK", "w_tenisK","f_Ks", "e_Ks", "w_Ks",\
                        "f_IRAC1", "e_IRAC1", "w_IRAC1", "f_IRAC2", "e_IRAC2", "w_IRAC2", "f_IRAC3", "e_IRAC3", "w_IRAC3", "f_IRAC4", "e_IRAC4", "w_IRAC4",\
                        "f_IA427", "e_IA427", "f_IA445", "e_IA445", "f_IA505", "e_IA505", "f_IA527", "e_IA527", "f_IA550", "e_IA550", "f_IA574", "e_IA574",\
                        "f_IA598", "e_IA598", "f_IA624", "e_IA624", "f_IA651", "e_IA651", "f_IA679", "e_IA679", "f_IA738", "e_IA738", "f_IA767", "e_IA767",\
                        "f_IA797", "e_IA797", "f_IA856", "e_IA856", "tot_cor", "wmin_ground", "wmin_hst","wmin_wfc3", "wmin_irac", "z_spec", "star_flag",\
                        "kron_radius","a_image", "b_image", "theta_J2000", "class_star", "flux_radius", "fwhm_image", "flags", "IRAC1_contam", "IRAC2_contam",\
                         "IRAC3_contam", "IRAC4_contam", "contam_flag","f140w_flag", "use_phot", "near_star", "nexp_f125w", "nexp_f140w", "nexp_f160w"]

df_fast = pd.read_csv('/home/siqi/goodss_3dhst.v4.1.cats/Fast/goodss_3dhst.v4.1.fout', delim_whitespace=True,header=None,comment='#',index_col=False)
df_fast.columns = ['id', 'z', 'ltau', 'metal','lage','Av','lmass','lsfr','lssfr','la2t','chi2']


# ###  Ma05
norm_wavelength= 5500.0
df_Ma = pd.read_csv('/home/siqi/M09_ssp_pickles.sed', delim_whitespace=True, header=None, comment='#',index_col=False)# only solar metallicity is contained in this catalogue
df_Ma.columns = ['Age','ZH','l','Flambda']
age = df_Ma.Age
metallicity = df_Ma.ZH
wavelength = df_Ma.l
Flux = df_Ma.Flambda
age_1Gyr_index = np.where(age==1.0)[0]
age_1Gyr = age[age_1Gyr_index]
metallicity_1Gyr = metallicity[age_1Gyr_index]
wavelength_1Gyr = wavelength[age_1Gyr_index]
Flux_1Gyr = Flux[age_1Gyr_index]
F_5500_1Gyr_index=np.where(wavelength_1Gyr==norm_wavelength)[0]
F_5500_1Gyr = Flux_1Gyr[wavelength_1Gyr==norm_wavelength].values # this is the band to be normalized 


df_M13 = pd.read_csv('/home/siqi/M13_models/sed_M13.ssz002',delim_whitespace=True,header=None,comment='#',index_col=False)
df_M13.columns = ['Age','ZH','l','Flambda']
age_M13 = df_M13.Age
metallicity_M13 = df_M13.ZH
wavelength_M13 = df_M13.l
Flux_M13 = df_M13.Flambda
age_1Gyr_index_M13 = np.where(age_M13==1.0)[0]#[0]
age_1Gyr_M13 = age_M13[age_1Gyr_index_M13]
metallicity_1Gyr_M13 = metallicity_M13[age_1Gyr_index_M13]
wavelength_1Gyr_M13 = wavelength_M13[age_1Gyr_index_M13]
Flux_1Gyr_M13 = Flux_M13[age_1Gyr_index_M13]
F_5500_1Gyr_index_M13=np.where(abs(wavelength_1Gyr_M13-norm_wavelength)<15)[0]
F_5500_1Gyr_M13 = 0.5*(Flux_1Gyr_M13.loc[62271+F_5500_1Gyr_index_M13[0]]+Flux_1Gyr_M13.loc[62271+F_5500_1Gyr_index_M13[1]])


# ### BC03
df_BC = pd.read_csv('/home/siqi/ssp_900Myr_z02.spec',delim_whitespace=True,header=None,comment='#',index_col=False)
df_BC.columns=['Lambda','Flux']
wavelength_BC = df_BC.Lambda
Flux_BC = df_BC.Flux
F_5500_BC_index=np.where(wavelength_BC==norm_wavelength)[0]
Flux_BC_norm = Flux_BC[F_5500_BC_index]

### Read in the BC03 models High-resolution, with Stelib library, Salpeter IMF, solar metallicity
BC03_fn='/home/siqi/bc03/models/Stelib_Atlas/Salpeter_IMF/bc2003_hr_stelib_m62_salp_ssp.ised_ASCII'
BC03_file = open(BC03_fn,"r")
BC03_X = []
for line in BC03_file:
    BC03_X.append(line)
BC03_SSP_m62 = np.array(BC03_X)
BC03_age_list = np.array(BC03_SSP_m62[0].split()[1:])
BC03_age_list_num = BC03_age_list.astype(np.float)/1.0e9 # unit is Gyr
BC03_wave_list = np.array(BC03_SSP_m62[6].split()[1:])
BC03_wave_list_num = BC03_wave_list.astype(np.float)
BC03_flux_list = np.array(BC03_SSP_m62[7:-12])
BC03_flux_array = np.zeros((221,7178))
for i in range(221):
    BC03_flux_array[i,:] = BC03_flux_list[i].split()[1:]
    BC03_flux_array[i,:] = BC03_flux_array[i,:]/BC03_flux_array[i,2556]# Normalize the flux


def read_spectra(row):
    """
    region: default 1 means the first region mentioned in the area, otherwise, the second region/third region
    """
    detector=df.detector[row]
    region = df.region[row]
    chip = df.chip[row]
    ID = df.ID[row]
    redshift_1=df_cat.loc[ID-1].z_best
    mag = -2.5*np.log10(df_cat.loc[ID-1].L161)+25#+0.02
    #print mag
    #WFC3 is using the infrared low-resolution grism, and here we are using the z band
    if detector == 'WFC3':
        filename="/home/siqi/GOODSS_WFC3_V4.1.5/goodss-"+"{0:02d}".format(region)+"/1D/ASCII/goodss-"+"{0:02d}".format(region)+"-G141_"+"{0:05d}".format(ID)+".1D.ascii"
        OneD_1 = np.loadtxt(filename,skiprows=1)
    if detector =="ACS":
        filename="/home/siqi/GOODSS_ACS_V4.1.5/acs-goodss-"+"{0:02d}".format(region)+"/1D/FITS/"+df.filename[row]
        OneD_1 = fits.getdata(filename, ext=1)
    return ID, OneD_1,redshift_1, mag
def Lick_index_ratio(wave, flux, band=3):
    if band == 3:
        blue_min = 1.06e4  # 1.072e4#
        blue_max = 1.08e4  # 1.08e4#
        red_min = 1.12e4  # 1.097e4#
        red_max = 1.14e4  # 1.106e4#
        band_min = blue_max
        band_max = red_min


    # Blue
    blue_mask = (wave >= blue_min) & (wave <= blue_max)
    blue_wave = wave[blue_mask]
    blue_flux = flux[blue_mask]

    # Red
    red_mask = (wave >= red_min) & (wave <= red_max)
    red_wave = wave[red_mask]
    red_flux = flux[red_mask]

    band_mask = (wave >= band_min) & (wave <= band_max)
    band_wave = wave[band_mask]
    band_flux = flux[band_mask]

    if len(blue_wave) == len(red_wave) and len(blue_wave) != 0:
        ratio = np.mean(blue_flux) / np.mean(red_flux)
    elif red_wave == []:
        ratio = np.mean(blue_flux) / np.mean(red_flux)
    elif len(blue_wave) != 0 and len(red_wave) != 0:
        ratio = np.mean(blue_flux) / np.mean(red_flux)

    # ratio_err = np.sqrt(np.sum(1/red_flux**2*blue_flux_err**2)+np.sum((blue_flux/red_flux**2*red_flux_err)**2))

    return ratio  # , ratio_err
def binning_spec_keep_shape(wave,flux,bin_size):
    wave_binned = wave
    flux_binned = np.zeros(len(wave))
    for i in range((int(len(wave)/bin_size))+1):
        flux_binned[bin_size*i:bin_size*(i+1)] = np.mean(flux[bin_size*i:bin_size*(i+1)])
    return wave_binned, flux_binned#, flux_err_binned
def derive_1D_spectra_Av_corrected(OneD_1, redshift_1, rownumber, wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod, A_v):
    """
    OneD_1 is the oneD spectra
    redshift_1 is the redshift of the spectra
    rownumber is the row number in order to store the spectra
    """
    region = df.region[rownumber]
    ID = df.ID[rownumber]
    n = len(OneD_1)
    age =10**(df_fast.loc[ID-1].lage)/1e9 ## in Gyr
    metal = df_fast.loc[ID-1].metal
    sfr = 10**(df_fast.loc[ID-1].lsfr)
    intrinsic_Av = df_fast.loc[ID-1].Av   
    
    norm_factor_BC = int((OneD_1[int(n/2+1)][0]-OneD_1[int(n/2)][0])/(1+redshift_1)/1)
    norm_limit_BC = int(5930/norm_factor_BC)*norm_factor_BC+400
    smooth_wavelength_BC_1 = wavelength_BC[400:norm_limit_BC].values.reshape(-1,norm_factor_BC).mean(axis=1)
    smooth_wavelength_BC = np.hstack([smooth_wavelength_BC_1,wavelength_BC[norm_limit_BC:]])

    smooth_Flux_BC_1 = Flux_BC[400:norm_limit_BC].values.reshape(-1,norm_factor_BC).mean(axis=1)
    smooth_Flux_BC = np.hstack([smooth_Flux_BC_1,Flux_BC[norm_limit_BC:]])/Flux_BC_norm.values[0]
    
    norm_factor_Ma = int((OneD_1[int(n/2+1)][0]-OneD_1[int(n/2)][0])/(1+redshift_1)/5)
    norm_limit_Ma = int(4770/norm_factor_Ma)*norm_factor_Ma
    smooth_wavelength_Ma = wavelength_1Gyr[:norm_limit_Ma].values.reshape(-1,norm_factor_Ma).mean(axis=1)
    smooth_Flux_Ma_1Gyr = Flux_1Gyr[:norm_limit_Ma].values.reshape(-1,norm_factor_Ma).mean(axis=1)/F_5500_1Gyr

    # Normalize the flux
    if redshift_1<=0.1:
        i = 12
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA574: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    elif redshift_1<=0.2:
        i = 13
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA624: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    elif redshift_1<=0.3:
        i = 15
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA679: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    elif redshift_1<=0.4:
        i = 16
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA738: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    elif redshift_1<=0.5:
        i = 18
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA797: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    elif redshift_1<=0.6:
        i = 19
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at IA856: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    else:
        i = 26
        temp_norm_wave = wave_list[i]#/(1+redshift_1)
        index_wave_norm = find_nearest(smooth_wavelength_BC,temp_norm_wave)
        norm_band = photometric_flux[i] 
        #plt.text(5000,0.55,'normalized at F850LPcand: rest frame '+"{0:.2f}".format(temp_norm_wave),fontsize=16)
        #plt.axvline(temp_norm_wave,linewidth=2,color='b')
    
    x = np.zeros(n)
    y = np.zeros(n)
    y_err = np.zeros(n)
    sensitivity = np.zeros(n)
    for i in range(0,n):
        x[i] = OneD_1[i][0]#/(1+redshift_1)
    print('wavelength range:',x[0],x[-1])
    spectra_extinction = calzetti00(x, A_v, 4.05)

    for i in range(n):
        spectra_flux_correction = 10**(0.4*spectra_extinction[i])# from obs to obtain the true value: the absolute value
        x[i] = x[i]/(1+redshift_1)
        y[i] = (OneD_1[i][1]-OneD_1[i][3])/OneD_1[i][6]*spectra_flux_correction#/Flux_0 # (flux-contamination)/sensitivity
        y_err[i] = OneD_1[i][2]/OneD_1[i][6]*spectra_flux_correction#/Flux_0
        sensitivity[i] = OneD_1[i][6]
    # end_index = np.argmin(np.diff(sensitivity[263:282],2)[1:],0)+263
    # start_index = np.argmin(np.diff(sensitivity[40:50],2)[1:])+42
    start_index = np.argmin(abs(x*(1+redshift_1)-11407.53))
    end_index = np.argmin(abs(x*(1+redshift_1)-16428.61))

    print('masking region:',x[start_index]*(1+redshift_1),x[end_index]*(1+redshift_1),start_index,end_index)
    # plt.plot(x*(1+redshift_1),sensitivity,color='k')
    # plt.plot(x[start_index:end_index]*(1+redshift_1),sensitivity[start_index:end_index],color='red')
    print('before masking',len(x))
    x = x[start_index:end_index]#[int(n*2/10):int(n*8/10)]
    y = y[start_index:end_index]*1e-17/norm_band#[int(n*2/10):int(n*8/10)]*1e-17/norm_band
    y_err = y_err[start_index:end_index]*1e-17/norm_band#[int(n*2/10):int(n*8/10)]*1e-17/norm_band
    print('after masking',len(x))
    
    # mask_non_neg_photo = np.where(photometric_flux>0)
    # wave_list = wave_list[mask_non_neg_photo]
    # band_list = band_list[mask_non_neg_photo]
    # photometric_flux = photometric_flux[mask_non_neg_photo]
    # photometric_flux_err = photometric_flux_err[mask_non_neg_photo]
    # photometric_flux_err_mod = photometric_flux_err_mod[mask_non_neg_photo]

    return x, y, y_err, wave_list/(1+redshift_1), band_list/(1+redshift_1), photometric_flux/norm_band, photometric_flux_err/norm_band, photometric_flux_err_mod/norm_band        


columns = ['ID','region','field',
          'M05_age_opt','M05_AV_opt','M13_age_opt','M13_AV_opt','BC_age_opt','BC_AV_opt',\
          'x2_spectra_M05_opt','x2_photo_M05_opt','x2_spectra_M13_opt','x2_photo_M13_opt','x2_spectra_BC_opt','x2_photo_BC_opt',\
          'M05_age_MCMC50','M05_age_std','M05_AV_MCMC50','M05_AV_std','M13_age_MCMC50','M13_age_std','M13_AV_MCMC50','M13_AV_std','BC_age_MCMC50','BC_age_std','BC_AV_MCMC50','BC_AV_std',\
          'x2_spectra_M05_MCMC50','x2_photo_M05_MCMC50','x2_spectra_M13_MCMC50','x2_photo_M13_MCMC50','x2_spectra_BC_MCMC50','x2_photo_BC_MCMC50',\
          'x2_M05_opt','x2_M13_opt','x2_BC_opt','x2_M05_MCMC50','x2_M13_MCMC50','x2_BC_MCMC50',\
          'model','grism_index','grism_index_AV_corr','age_opt','age_opt_std','AV_opt','AV_opt_std']
chi_square_list = pd.DataFrame(index=df.index,columns=columns)
chi_square_list_final = pd.DataFrame(index=df.index,columns=columns)

weight1 = 1#0.25#0.0#1/1.864#/0.5
weight2 = 1#weight1*5e-3#1.0#1/1228.53#/0.5

## Prepare the M05 models and store in the right place
M05_model = []
M05_model_list=[]
for i in range(30):
    age_index = i
    age_prior = df_Ma.Age.unique()[age_index]
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')
    fn1 = '/home/siqi/SSP_models/new/M05_age_'+'0_'+split_galaxy_age_string[1]+'_Av_00_z002.csv'
    M05_model = np.loadtxt(fn1)
    M05_model_list.append(M05_model)
fn1 = '/home/siqi/SSP_models/new/M05_age_1_Av_00_z002.csv'
fn2 = '/home/siqi/SSP_models/new/M05_age_1_5_Av_00_z002.csv'
M05_model = np.loadtxt(fn1)
M05_model_list.append(M05_model)
M05_model = np.loadtxt(fn2)
M05_model_list.append(M05_model)
for i in range(32,46):
    age_index = i
    age_prior = df_Ma.Age.unique()[age_index]
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')
    fn2 = '/home/siqi/SSP_models/new/M05_age_'+split_galaxy_age_string[0]+'_Av_00_z002.csv'
    M05_model = np.loadtxt(fn2)
    M05_model_list.append(M05_model)


## Prepare the M13 models and store in the right place
M13_model = []
M13_model_list=[]
fn1 = '/home/siqi/SSP_models/new/M13_age_1e-06_Av_00_z002.csv'
fn2 = '/home/siqi/SSP_models/new/M13_age_0_0001_Av_00_z002.csv'
M13_model = np.genfromtxt(fn1)
M13_model_list.append(M13_model)
M13_model = np.genfromtxt(fn2)
M13_model_list.append(M13_model)
for i in range(2,51):
    age_index = i
    age_prior = df_M13.Age.unique()[age_index]
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')
    fn1 = '/home/siqi/SSP_models/new/M13_age_'+'0_'+split_galaxy_age_string[1]+'_Av_00_z002.csv'
    M13_model = np.loadtxt(fn1)
    M13_model_list.append(M13_model)
fn1 = '/home/siqi/SSP_models/new/M13_age_1_Av_00_z002.csv'
fn2 = '/home/siqi/SSP_models/new/M13_age_1_5_Av_00_z002.csv'
M13_model = np.loadtxt(fn1)
M13_model_list.append(M13_model)
M13_model = np.loadtxt(fn2)
M13_model_list.append(M13_model)
for i in range(53,67):
    age_index = i
    age_prior = df_M13.Age.unique()[age_index]
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')
    fn2 = '/home/siqi/SSP_models/new/M13_age_'+split_galaxy_age_string[0]+'_Av_00_z002.csv'
    M13_model = np.loadtxt(fn2)
    M13_model_list.append(M13_model)

def binning_spec_keep_shape_x(wave,flux,flux_err,bin_size):
    wave_binned = wave
    flux_binned = np.zeros(len(wave))
    flux_err_binned = np.zeros(len(wave))
    for i in range((int(len(wave)/bin_size))+1):
        flux_binned[bin_size*i:bin_size*(i+1)] = np.mean(flux[bin_size*i:bin_size*(i+1)])
        flux_err_binned[bin_size*i:bin_size*(i+1)] = np.mean(flux_err[bin_size*i:bin_size*(i+1)])
    return wave_binned, flux_binned, flux_err_binned

def minimize_age_AV_vector_weighted(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    # print('minimize process age av grid',X)

    n=len(x)
    age_index = find_nearest(df_Ma.Age.unique(), galaxy_age)
    age_prior = df_Ma.Age.unique()[age_index]
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')
    # print(age_prior)
    
    if age_prior < 1:
        if galaxy_age < age_prior:
            model1 = (M05_model_list[age_index]*(galaxy_age-df_Ma.Age.unique()[age_index-1]) \
                + M05_model_list[age_index-1]*(age_prior-galaxy_age))/(df_Ma.Age.unique()[age_index]-df_Ma.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model1 = (M05_model_list[age_index]*(df_Ma.Age.unique()[age_index+1]-galaxy_age) \
                + M05_model_list[age_index+1]*(galaxy_age-age_prior))/(df_Ma.Age.unique()[age_index+1]-df_Ma.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model1 = M05_model_list[age_index]
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model1 = (3.0-galaxy_age)*M05_model_list[32] + (galaxy_age-2.0)*M05_model_list[33]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model1 = (4.0-galaxy_age)*M05_model_list[33] + (galaxy_age-3.0)*M05_model_list[34]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model1 = (5.0-galaxy_age)*M05_model_list[34] + (galaxy_age-4.0)*M05_model_list[35]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model1 = (6.0-galaxy_age)*M05_model_list[35] + (galaxy_age-5.0)*M05_model_list[36]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model1 = (7.0-galaxy_age)*M05_model_list[36] + (galaxy_age-6.0)*M05_model_list[37]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model1 = (8.0-galaxy_age)*M05_model_list[37] + (galaxy_age-7.0)*M05_model_list[38]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model1 = (9.0-galaxy_age)*M05_model_list[38] + (galaxy_age-8.0)*M05_model_list[39]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model1 = (10.0-galaxy_age)*M05_model_list[39] + (galaxy_age-9.0)*M05_model_list[40]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model1 = (11.0-galaxy_age)*M05_model_list[40] + (galaxy_age-10.0)*M05_model_list[41]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model1 = (12.0-galaxy_age)*M05_model_list[41] + (galaxy_age-11.0)*M05_model_list[42]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model1 = (13.0-galaxy_age)*M05_model_list[42] + (galaxy_age-12.0)*M05_model_list[43]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model1 = (14.0-galaxy_age)*M05_model_list[43] + (galaxy_age-13.0)*M05_model_list[44]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model1 = (15.0-galaxy_age)*M05_model_list[44] + (galaxy_age-14.0)*M05_model_list[45]
        else:
            model1 = M05_model_list[age_index]

    spectra_extinction = calzetti00(model1[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M05_flux_center = model1[1,:]*spectra_flux_correction
    F_M05_index=700#167
    Flux_M05_norm_new = M05_flux_center[F_M05_index]
    smooth_Flux_Ma_1Gyr_new = M05_flux_center/Flux_M05_norm_new

    binning_index = find_nearest(model1[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model1[0,binning_index]-model1[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model1[0,binning_index]-model1[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model1[0,:], smooth_Flux_Ma_1Gyr_new,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        if np.isnan(x2):
            print('spectra chi2 is nan,binning model',model_flux_binned)
            print('spectra model wave', model1[0,:], model1[1,:], intrinsic_Av)
            print('model flux before binning', spectra_extinction, spectra_flux_correction, M05_flux_center, Flux_M05_norm_new)
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1, wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning model, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    else:
        binning_size = int((model1[0,binning_index]-model1[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model1[0,:], smooth_Flux_Ma_1Gyr_new) 
        # print('binning data, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
        x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new, redshift_1, wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
    
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            x2_tot = 0.5*weight1*x2+0.5*weight2*x2_photo
        else:
            x2_tot = np.inf
    except ValueError: # NaN value case
        x2_tot = np.inf
        print('ValueError', x2_tot)
    # print('M05 x2 tot:',x2, x2_photo, x2_tot)
    return x2_tot
def lg_minimize_age_AV_vector_weighted(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_Ma.Age.unique(), galaxy_age)
    age_prior = df_Ma.Age.unique()[age_index]
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    if age_prior < 1:
        if galaxy_age < age_prior:
            model1 = (M05_model_list[age_index]*(galaxy_age-df_Ma.Age.unique()[age_index-1]) \
                + M05_model_list[age_index-1]*(age_prior-galaxy_age))/(df_Ma.Age.unique()[age_index]-df_Ma.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model1 = (M05_model_list[age_index]*(df_Ma.Age.unique()[age_index+1]-galaxy_age) \
                + M05_model_list[age_index+1]*(galaxy_age-age_prior))/(df_Ma.Age.unique()[age_index+1]-df_Ma.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model1 = M05_model_list[age_index]
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model1 = (3.0-galaxy_age)*M05_model_list[32] + (galaxy_age-2.0)*M05_model_list[33]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model1 = (4.0-galaxy_age)*M05_model_list[33] + (galaxy_age-3.0)*M05_model_list[34]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model1 = (5.0-galaxy_age)*M05_model_list[34] + (galaxy_age-4.0)*M05_model_list[35]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model1 = (6.0-galaxy_age)*M05_model_list[35] + (galaxy_age-5.0)*M05_model_list[36]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model1 = (7.0-galaxy_age)*M05_model_list[36] + (galaxy_age-6.0)*M05_model_list[37]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model1 = (8.0-galaxy_age)*M05_model_list[37] + (galaxy_age-7.0)*M05_model_list[38]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model1 = (9.0-galaxy_age)*M05_model_list[38] + (galaxy_age-8.0)*M05_model_list[39]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model1 = (10.0-galaxy_age)*M05_model_list[39] + (galaxy_age-9.0)*M05_model_list[40]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model1 = (11.0-galaxy_age)*M05_model_list[40] + (galaxy_age-10.0)*M05_model_list[41]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model1 = (12.0-galaxy_age)*M05_model_list[41] + (galaxy_age-11.0)*M05_model_list[42]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model1 = (13.0-galaxy_age)*M05_model_list[42] + (galaxy_age-12.0)*M05_model_list[43]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model1 = (14.0-galaxy_age)*M05_model_list[43] + (galaxy_age-13.0)*M05_model_list[44]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model1 = (15.0-galaxy_age)*M05_model_list[44] + (galaxy_age-14.0)*M05_model_list[45]
        else:
            model1 = M05_model_list[age_index]

    spectra_extinction = calzetti00(model1[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M05_flux_center = model1[1,:]*spectra_flux_correction
    F_M05_index=700#167
    Flux_M05_norm_new = M05_flux_center[F_M05_index]
    smooth_Flux_Ma_1Gyr_new = M05_flux_center/Flux_M05_norm_new

    binning_index = find_nearest(model1[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model1[0,binning_index]-model1[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model1[0,binning_index]-model1[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model1[0,:], smooth_Flux_Ma_1Gyr_new,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        # x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning model, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    else:
        binning_size = int((model1[0,binning_index]-model1[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model1[0,:], smooth_Flux_Ma_1Gyr_new) 
        x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning data, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    # print('binning size, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, model1[0,:], smooth_Flux_Ma_1Gyr_new)
    
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            lnprobval = -0.5*(0.5*x2+0.5*x2_photo)#np.log(np.exp(-0.5*(0.5*weight1*x2+0.5*weight2*x2_photo)))
            if np.isnan(lnprobval):
                lnprobval = -np.inf
        else:
            lnprobval = -np.inf
    except ValueError: # NaN value case
       lnprobval = -np.inf
       print('valueError',lnprobval)
    # print('lnprob:',lnprobval, x2, x2_photo)
    return lnprobval
def minimize_age_AV_vector_weighted_return_flux(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_Ma.Age.unique(), galaxy_age)
    age_prior = df_Ma.Age.unique()[age_index]
    #print('galaxy age', galaxy_age, 'age prior:', age_prior)
    AV_string = str(intrinsic_Av)
    #print('intrinsic Av:', intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    if age_prior < 1:
        if galaxy_age < age_prior:
            model1 = (M05_model_list[age_index]*(galaxy_age-df_Ma.Age.unique()[age_index-1]) \
                + M05_model_list[age_index-1]*(age_prior-galaxy_age))/(df_Ma.Age.unique()[age_index]-df_Ma.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model1 = (M05_model_list[age_index]*(df_Ma.Age.unique()[age_index+1]-galaxy_age) \
                + M05_model_list[age_index+1]*(galaxy_age-age_prior))/(df_Ma.Age.unique()[age_index+1]-df_Ma.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model1 = M05_model_list[age_index]
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model1 = (3.0-galaxy_age)*M05_model_list[32] + (galaxy_age-2.0)*M05_model_list[33]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model1 = (4.0-galaxy_age)*M05_model_list[33] + (galaxy_age-3.0)*M05_model_list[34]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model1 = (5.0-galaxy_age)*M05_model_list[34] + (galaxy_age-4.0)*M05_model_list[35]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model1 = (6.0-galaxy_age)*M05_model_list[35] + (galaxy_age-5.0)*M05_model_list[36]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model1 = (7.0-galaxy_age)*M05_model_list[36] + (galaxy_age-6.0)*M05_model_list[37]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model1 = (8.0-galaxy_age)*M05_model_list[37] + (galaxy_age-7.0)*M05_model_list[38]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model1 = (9.0-galaxy_age)*M05_model_list[38] + (galaxy_age-8.0)*M05_model_list[39]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model1 = (10.0-galaxy_age)*M05_model_list[39] + (galaxy_age-9.0)*M05_model_list[40]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model1 = (11.0-galaxy_age)*M05_model_list[40] + (galaxy_age-10.0)*M05_model_list[41]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model1 = (12.0-galaxy_age)*M05_model_list[41] + (galaxy_age-11.0)*M05_model_list[42]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model1 = (13.0-galaxy_age)*M05_model_list[42] + (galaxy_age-12.0)*M05_model_list[43]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model1 = (14.0-galaxy_age)*M05_model_list[43] + (galaxy_age-13.0)*M05_model_list[44]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model1 = (15.0-galaxy_age)*M05_model_list[44] + (galaxy_age-14.0)*M05_model_list[45]
        else:
            model1 = M05_model_list[age_index]
    
    spectra_extinction = calzetti00(model1[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M05_flux_center = model1[1,:]*spectra_flux_correction
    F_M05_index=700#167
    Flux_M05_norm_new = M05_flux_center[F_M05_index]
    smooth_Flux_Ma_1Gyr_new = M05_flux_center/Flux_M05_norm_new

    binning_index = find_nearest(model1[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model1[0,binning_index]-model1[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model1[0,binning_index]-model1[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model1[0,:], smooth_Flux_Ma_1Gyr_new,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        # x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # print('binning model, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    else:
        binning_size = int((model1[0,binning_index]-model1[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model1[0,:], smooth_Flux_Ma_1Gyr_new) 
        x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning data, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, model1[0,:], smooth_Flux_Ma_1Gyr_new)
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            x2_tot = 0.5*weight1*x2+0.5*weight2*x2_photo
        else:
            x2_tot = np.inf
    except ValueError: # NaN value case
       x2_tot = np.inf
       print('valueError', x2_tot)
    # print('model wave range', model1[0,0], model1[0,-1])

    return x2_tot, model1[0,:], smooth_Flux_Ma_1Gyr_new
def minimize_age_AV_vector_weighted_return_chi2_sep(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_Ma.Age.unique(), galaxy_age)
    age_prior = df_Ma.Age.unique()[age_index]
    #print('galaxy age', galaxy_age, 'age prior:', age_prior)
    AV_string = str(intrinsic_Av)
    #print('intrinsic Av:', intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    if age_prior < 1:
        if galaxy_age < age_prior:
            model1 = (M05_model_list[age_index]*(galaxy_age-df_Ma.Age.unique()[age_index-1]) \
                + M05_model_list[age_index-1]*(age_prior-galaxy_age))/(df_Ma.Age.unique()[age_index]-df_Ma.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model1 = (M05_model_list[age_index]*(df_Ma.Age.unique()[age_index+1]-galaxy_age) \
                + M05_model_list[age_index+1]*(galaxy_age-age_prior))/(df_Ma.Age.unique()[age_index+1]-df_Ma.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model1 = M05_model_list[age_index]
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model1 = 2.*(1.5-galaxy_age)*M05_model_list[30] + 2.*(galaxy_age-1.0)*M05_model_list[31]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model1 = 2.*(2.0-galaxy_age)*M05_model_list[31] + 2.*(galaxy_age-1.5)*M05_model_list[32]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model1 = (3.0-galaxy_age)*M05_model_list[32] + (galaxy_age-2.0)*M05_model_list[33]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model1 = (4.0-galaxy_age)*M05_model_list[33] + (galaxy_age-3.0)*M05_model_list[34]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model1 = (5.0-galaxy_age)*M05_model_list[34] + (galaxy_age-4.0)*M05_model_list[35]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model1 = (6.0-galaxy_age)*M05_model_list[35] + (galaxy_age-5.0)*M05_model_list[36]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model1 = (7.0-galaxy_age)*M05_model_list[36] + (galaxy_age-6.0)*M05_model_list[37]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model1 = (8.0-galaxy_age)*M05_model_list[37] + (galaxy_age-7.0)*M05_model_list[38]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model1 = (9.0-galaxy_age)*M05_model_list[38] + (galaxy_age-8.0)*M05_model_list[39]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model1 = (10.0-galaxy_age)*M05_model_list[39] + (galaxy_age-9.0)*M05_model_list[40]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model1 = (11.0-galaxy_age)*M05_model_list[40] + (galaxy_age-10.0)*M05_model_list[41]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model1 = (12.0-galaxy_age)*M05_model_list[41] + (galaxy_age-11.0)*M05_model_list[42]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model1 = (13.0-galaxy_age)*M05_model_list[42] + (galaxy_age-12.0)*M05_model_list[43]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model1 = (14.0-galaxy_age)*M05_model_list[43] + (galaxy_age-13.0)*M05_model_list[44]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model1 = (15.0-galaxy_age)*M05_model_list[44] + (galaxy_age-14.0)*M05_model_list[45]
        else:
            model1 = M05_model_list[age_index]

    spectra_extinction = calzetti00(model1[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M05_flux_center = model1[1,:]*spectra_flux_correction
    F_M05_index=700#167
    Flux_M05_norm_new = M05_flux_center[F_M05_index]
    smooth_Flux_Ma_1Gyr_new = M05_flux_center/Flux_M05_norm_new
    
    binning_index = find_nearest(model1[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model1[0,binning_index]-model1[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model1[0,binning_index]-model1[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model1[0,:], smooth_Flux_Ma_1Gyr_new,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning model, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    else:
        binning_size = int((model1[0,binning_index]-model1[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model1[0,:], smooth_Flux_Ma_1Gyr_new) 
        x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # x2_photo = chisquare_photo(model1[0,:], smooth_Flux_Ma_1Gyr_new, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning data, model 1', n, (model1[0,binning_index]-model1[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, model1[0,:], smooth_Flux_Ma_1Gyr_new)
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            pass
        else:
            x2 = np.inf
            x2_photo = np.inf
    except ValueError: # NaN value case
       x2 = np.inf
       x2_photo = np.inf
       print('ValueError', x2)
    return x2, x2_photo

def minimize_age_AV_vector_weighted_M13(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    # print('minimize process age av grid M13:',X)

    n=len(x)
    age_index = find_nearest(df_M13.Age.unique(), galaxy_age)
    age_prior = df_M13.Age.unique()[age_index]
    age_prior = float(age_prior)
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    if age_prior < 1e-5:
        model2 = M13_model_list[0]
    elif age_prior >= 1e-5 and age_prior < 1:
        if galaxy_age < age_prior:
            model2 = (M13_model_list[age_index]*(galaxy_age-df_M13.Age.unique()[age_index-1]) \
                + M13_model_list[age_index-1]*(age_prior-galaxy_age))/(df_M13.Age.unique()[age_index]-df_M13.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model2 = (M13_model_list[age_index]*(df_M13.Age.unique()[age_index+1]-galaxy_age) \
                + M13_model_list[age_index+1]*(galaxy_age-age_prior))/(df_M13.Age.unique()[age_index+1]-df_M13.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model2 = M13_model_list[age_index] 
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model2 = (3.0-galaxy_age)*M13_model_list[53] + (galaxy_age-2.0)*M13_model_list[54]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model2 = (4.0-galaxy_age)*M13_model_list[54] + (galaxy_age-3.0)*M13_model_list[55]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model2 = (5.0-galaxy_age)*M13_model_list[55] + (galaxy_age-4.0)*M13_model_list[56]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model2 = (6.0-galaxy_age)*M13_model_list[56] + (galaxy_age-5.0)*M13_model_list[57]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model2 = (7.0-galaxy_age)*M13_model_list[57] + (galaxy_age-6.0)*M13_model_list[58]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model2 = (8.0-galaxy_age)*M13_model_list[58] + (galaxy_age-7.0)*M13_model_list[59]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model2 = (9.0-galaxy_age)*M13_model_list[59] + (galaxy_age-8.0)*M13_model_list[60]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model2 = (10.0-galaxy_age)*M13_model_list[60] + (galaxy_age-9.0)*M13_model_list[61]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model2 = (11.0-galaxy_age)*M13_model_list[61] + (galaxy_age-10.0)*M13_model_list[62]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model2 = (12.0-galaxy_age)*M13_model_list[62] + (galaxy_age-11.0)*M13_model_list[63]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model2 = (13.0-galaxy_age)*M13_model_list[63] + (galaxy_age-12.0)*M13_model_list[64]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model2 = (14.0-galaxy_age)*M13_model_list[64] + (galaxy_age-13.0)*M13_model_list[65]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model2 = (15.0-galaxy_age)*M13_model_list[65] + (galaxy_age-14.0)*M13_model_list[66]
        else:
            model2 = M13_model_list[age_index]

    spectra_extinction = calzetti00(model2[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M13_flux_center = model2[1,:]*spectra_flux_correction
    F_M13_index = 326#126##np.where(abs(model2[0,:]-norm_wavelength)<10.5)[0][0]
    Flux_M13_norm_new = M13_flux_center[F_M13_index]
    smooth_Flux_M13_1Gyr_new = M13_flux_center/Flux_M13_norm_new

    binning_index = find_nearest(model2[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model2[0,binning_index]-model2[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model2[0,binning_index]-model2[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model2[0,:], smooth_Flux_M13_1Gyr_new,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        if np.isnan(x2):
            print('spectra chi2 is nan, binning model', model_flux_binned)
            print('spectra model wave', model2[0,:],intrinsic_Av)
            print('model flux before binning', spectra_extinction, spectra_flux_correction, M13_flux_center, Flux_M13_norm_new)
            sys.exit()
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning model, model 2', n, (model2[0,binning_index]-model2[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]),binning_size)    
    else:
        binning_size = int((model2[0,binning_index]-model2[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned = binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model2[0,:], smooth_Flux_M13_1Gyr_new) 
        if np.isnan(x2):
            print('spectra chi2 is nan,binning data',x_binned)
            print('spectra model wave', model2[0,:],intrinsic_Av)
            print('model flux before binning', spectra_extinction, spectra_flux_correction, M13_flux_center, Flux_M13_norm_new)
            sys.exit()
        x2_photo = chisquare_photo(model2[0,:], smooth_Flux_M13_1Gyr_new,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        if np.isnan(x2_photo):
            print('model 2 photo nan', x2_photo)
        # print('binning data, model 2', n, (model2[0,binning_index]-model2[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]),binning_size)    
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, model2[0,:], smooth_Flux_M13_1Gyr_new)
    # print(x2_photo)
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            x2_tot = 0.5*weight1*x2+0.5*weight2*x2_photo
        else:
            x2_tot = np.inf
    except ValueError: # NaN value case
       x2_tot = np.inf
       print('ValueError', x2_tot)
    return x2_tot
def lg_minimize_age_AV_vector_weighted_M13(X):
    tik = time.clock()
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_M13.Age.unique(), galaxy_age)
    age_prior = df_M13.Age.unique()[age_index]
    age_prior = float(age_prior)
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    model2 = np.zeros((2,762))
    if age_prior < 1e-5:
        model2 = M13_model_list[0]
    elif age_prior >= 1e-5 and age_prior < 1:
        if galaxy_age < age_prior:
            model2 = (M13_model_list[age_index]*(galaxy_age-df_M13.Age.unique()[age_index-1]) \
                + M13_model_list[age_index-1]*(age_prior-galaxy_age))/(df_M13.Age.unique()[age_index]-df_M13.Age.unique()[age_index-1])
            # print('age interval', (galaxy_age-df_M13.Age.unique()[age_index-1]), (age_prior-galaxy_age))
        elif galaxy_age > age_prior:
            model2 = (M13_model_list[age_index]*(df_M13.Age.unique()[age_index+1]-galaxy_age) \
                + M13_model_list[age_index+1]*(galaxy_age-age_prior))/(df_M13.Age.unique()[age_index+1]-df_M13.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model2 = M13_model_list[age_index] 
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model2 = (3.0-galaxy_age)*M13_model_list[53] + (galaxy_age-2.0)*M13_model_list[54]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model2 = (4.0-galaxy_age)*M13_model_list[54] + (galaxy_age-3.0)*M13_model_list[55]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model2 = (5.0-galaxy_age)*M13_model_list[55] + (galaxy_age-4.0)*M13_model_list[56]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model2 = (6.0-galaxy_age)*M13_model_list[56] + (galaxy_age-5.0)*M13_model_list[57]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model2 = (7.0-galaxy_age)*M13_model_list[57] + (galaxy_age-6.0)*M13_model_list[58]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model2 = (8.0-galaxy_age)*M13_model_list[58] + (galaxy_age-7.0)*M13_model_list[59]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model2 = (9.0-galaxy_age)*M13_model_list[59] + (galaxy_age-8.0)*M13_model_list[60]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model2 = (10.0-galaxy_age)*M13_model_list[60] + (galaxy_age-9.0)*M13_model_list[61]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model2 = (11.0-galaxy_age)*M13_model_list[61] + (galaxy_age-10.0)*M13_model_list[62]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model2 = (12.0-galaxy_age)*M13_model_list[62] + (galaxy_age-11.0)*M13_model_list[63]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model2 = (13.0-galaxy_age)*M13_model_list[63] + (galaxy_age-12.0)*M13_model_list[64]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model2 = (14.0-galaxy_age)*M13_model_list[64] + (galaxy_age-13.0)*M13_model_list[65]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model2 = (15.0-galaxy_age)*M13_model_list[65] + (galaxy_age-14.0)*M13_model_list[66]
        else:
            model2 = M13_model_list[age_index]


    spectra_extinction = calzetti00(model2[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M13_flux_center = model2[1,:]*spectra_flux_correction
    F_M13_index = 326#126##np.where(abs(model2[0,:]-norm_wavelength)<10.5)[0][0]
    Flux_M13_norm_new = M13_flux_center[F_M13_index]
    smooth_Flux_M13_1Gyr_new = M13_flux_center/Flux_M13_norm_new
    
    binning_index = find_nearest(model2[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index == len(model2[0,:]):
        binning_index = len(model2[0,:])-1
        # print('binning index:',binning_index,len(model2[0,:]),len(x), model2[:,binning_index-2:binning_index])

    # print('galaxy age:', galaxy_age, age_prior,age_index)
    # print(x, n)
    # print(len(model2),galaxy_age, age_prior, age_index, len(x), len(model2), np.median(x), np.min(model2[0,:]),np.max(model2[0,:]), binning_index)
    if (x[int(n/2)]-x[int(n/2)-1]) > (model2[0,binning_index]-model2[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model2[0,binning_index]-model2[0,binning_index-1]))
        # print('bin size', model2[0,binning_index],\
        #                   model2[0,binning_index-1],\
        #                   (model2[0,binning_index]-model2[0,binning_index-1]),\
        #                   int((x[int(n/2)]-x[int(n/2)-1])),\
        #                   binning_size)
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model2[0,:], smooth_Flux_M13_1Gyr_new, binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
    else:
        binning_size = int((model2[0,binning_index]-model2[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned = binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model2[0,:], smooth_Flux_M13_1Gyr_new) 
        x2_photo = chisquare_photo(model2[0,:], smooth_Flux_M13_1Gyr_new,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
    tok = time.clock()
    # print('time for lg_minimize',tok-tik)
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            lnprobval = -0.5*(0.5*x2+0.5*x2_photo)#np.log(np.exp(-0.5*(0.5*weight1*x2+0.5*weight2*x2_photo)))
            if np.isnan(lnprobval):
                lnprobval = -np.inf
        else:
            lnprobval = -np.inf
    except ValueError: # NaN value case
       lnprobval = -np.inf
       print('valueError',lnprobval,x2, x2_photo)
    # print('lnprob:',lnprobval)
    return lnprobval
def minimize_age_AV_vector_weighted_M13_return_flux(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_M13.Age.unique(), galaxy_age)
    age_prior = df_M13.Age.unique()[age_index]
    age_prior = float(age_prior)
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    model2 = np.zeros((2,762))
    if age_prior < 1e-5:
        model2 = M13_model_list[0]
    elif age_prior >= 1e-5 and age_prior < 1:
        if galaxy_age < age_prior:
            model2 = (M13_model_list[age_index]*(galaxy_age-df_M13.Age.unique()[age_index-1]) \
                + M13_model_list[age_index-1]*(age_prior-galaxy_age))/(df_M13.Age.unique()[age_index]-df_M13.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model2 = (M13_model_list[age_index]*(df_M13.Age.unique()[age_index+1]-galaxy_age) \
                + M13_model_list[age_index+1]*(galaxy_age-age_prior))/(df_M13.Age.unique()[age_index+1]-df_M13.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model2 = M13_model_list[age_index] 
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model2[0,:] = (3.0-galaxy_age)*M13_model_list[53][0,:] + (galaxy_age-2.0)*M13_model_list[54][0,:]
            model2[1,:] = (3.0-galaxy_age)*M13_model_list[53][1,:] + (galaxy_age-2.0)*M13_model_list[54][1,:]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model2 = (4.0-galaxy_age)*M13_model_list[54] + (galaxy_age-3.0)*M13_model_list[55]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model2 = (5.0-galaxy_age)*M13_model_list[55] + (galaxy_age-4.0)*M13_model_list[56]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model2 = (6.0-galaxy_age)*M13_model_list[56] + (galaxy_age-5.0)*M13_model_list[57]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model2 = (7.0-galaxy_age)*M13_model_list[57] + (galaxy_age-6.0)*M13_model_list[58]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model2 = (8.0-galaxy_age)*M13_model_list[58] + (galaxy_age-7.0)*M13_model_list[59]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model2 = (9.0-galaxy_age)*M13_model_list[59] + (galaxy_age-8.0)*M13_model_list[60]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model2 = (10.0-galaxy_age)*M13_model_list[60] + (galaxy_age-9.0)*M13_model_list[61]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model2 = (11.0-galaxy_age)*M13_model_list[61] + (galaxy_age-10.0)*M13_model_list[62]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model2 = (12.0-galaxy_age)*M13_model_list[62] + (galaxy_age-11.0)*M13_model_list[63]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model2 = (13.0-galaxy_age)*M13_model_list[63] + (galaxy_age-12.0)*M13_model_list[64]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model2 = (14.0-galaxy_age)*M13_model_list[64] + (galaxy_age-13.0)*M13_model_list[65]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model2 = (15.0-galaxy_age)*M13_model_list[65] + (galaxy_age-14.0)*M13_model_list[66]
        else:
            model2 = M13_model_list[age_index]

    
    spectra_extinction = calzetti00(model2[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M13_flux_center = model2[1,:]*spectra_flux_correction
    F_M13_index = 326#126##np.where(abs(model2[0,:]-norm_wavelength)<10.5)[0][0]
    Flux_M13_norm_new = M13_flux_center[F_M13_index]
    smooth_Flux_M13_1Gyr_new = M13_flux_center/Flux_M13_norm_new

    binning_index = find_nearest(model2[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model2[0,binning_index]-model2[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model2[0,binning_index]-model2[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model2[0,:], smooth_Flux_M13_1Gyr_new, binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        smooth_Flux_M13_1Gyr_new = model_flux_binned
    else:
        binning_size = int((model2[0,binning_index]-model2[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned = binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model2[0,:], smooth_Flux_M13_1Gyr_new) 
        x2_photo = chisquare_photo(model2[0,:], smooth_Flux_M13_1Gyr_new, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            x2_tot = 0.5*weight1*x2+0.5*weight2*x2_photo
        else:
            x2_tot = np.inf
    except ValueError: # NaN value case
       x2_tot = np.inf
       print('valueError', x2_tot)
    # print('model wave range', model2[0,0], model2[0,-1], split_galaxy_age_string )
    # print('model wave separately', M13_model_list[53][0,0],M13_model_list[53][0,-1],len(M13_model_list[53][0,:]),len(M13_model_list[54][0,:]),M13_model_list[54][0,0],M13_model_list[53][0,-1])
    # print('model test', model_test[0,0], model_test[0,-1])
    # print('age',galaxy_age,age_prior)
    return x2_tot, model2[0,:], smooth_Flux_M13_1Gyr_new
def minimize_age_AV_vector_weighted_M13_return_chi2_sep(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(df_M13.Age.unique(), galaxy_age)
    age_prior = df_M13.Age.unique()[age_index]
    age_prior = float(age_prior)
    AV_string = str(intrinsic_Av)
    galaxy_age_string = str(age_prior)
    split_galaxy_age_string = str(galaxy_age_string).split('.')

    if age_prior < 1e-5:
        model2 = M13_model_list[0]
    elif age_prior >= 1e-5 and age_prior < 1:
        if galaxy_age < age_prior:
            model2 = (M13_model_list[age_index]*(galaxy_age-df_M13.Age.unique()[age_index-1]) \
                + M13_model_list[age_index-1]*(age_prior-galaxy_age))/(df_M13.Age.unique()[age_index]-df_M13.Age.unique()[age_index-1])
        elif galaxy_age > age_prior:
            model2 = (M13_model_list[age_index]*(df_M13.Age.unique()[age_index+1]-galaxy_age) \
                + M13_model_list[age_index+1]*(galaxy_age-age_prior))/(df_M13.Age.unique()[age_index+1]-df_M13.Age.unique()[age_index])
        elif galaxy_age == age_prior:
            model2 = M13_model_list[age_index] 
    elif age_prior == 1.5:
        if galaxy_age >=1.25 and galaxy_age <1.5:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >= 1.5 and galaxy_age <= 1.75:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
    elif len(split_galaxy_age_string[1])==1:
        if galaxy_age >= 1.0 and galaxy_age < 1.25:
            model2 = 2.*(1.5-galaxy_age)*M13_model_list[51] + 2.*(galaxy_age-1.0)*M13_model_list[52]
        elif galaxy_age >=1.75 and galaxy_age < 2.0:
            model2 = 2.*(2.0-galaxy_age)*M13_model_list[52] + 2.*(galaxy_age-1.5)*M13_model_list[53]
        elif galaxy_age >= 2.0 and galaxy_age < 3.0:
            model2 = (3.0-galaxy_age)*M13_model_list[53] + (galaxy_age-2.0)*M13_model_list[54]
        elif galaxy_age >= 3.0 and galaxy_age < 4.0:
            model2 = (4.0-galaxy_age)*M13_model_list[54] + (galaxy_age-3.0)*M13_model_list[55]
        elif galaxy_age >= 4.0 and galaxy_age < 5.0:
            model2 = (5.0-galaxy_age)*M13_model_list[55] + (galaxy_age-4.0)*M13_model_list[56]
        elif galaxy_age >= 5.0 and galaxy_age < 6.0:
            model2 = (6.0-galaxy_age)*M13_model_list[56] + (galaxy_age-5.0)*M13_model_list[57]
        elif galaxy_age >= 6.0 and galaxy_age < 7.0:
            model2 = (7.0-galaxy_age)*M13_model_list[57] + (galaxy_age-6.0)*M13_model_list[58]
        elif galaxy_age >= 7.0 and galaxy_age < 8.0:
            model2 = (8.0-galaxy_age)*M13_model_list[58] + (galaxy_age-7.0)*M13_model_list[59]
        elif galaxy_age >= 8.0 and galaxy_age < 9.0:
            model2 = (9.0-galaxy_age)*M13_model_list[59] + (galaxy_age-8.0)*M13_model_list[60]
        elif galaxy_age >= 9.0 and galaxy_age < 10.0:
            model2 = (10.0-galaxy_age)*M13_model_list[60] + 2.*(galaxy_age-9.0)*M13_model_list[61]
        elif galaxy_age >= 10.0 and galaxy_age < 11.0:
            model2 = (11.0-galaxy_age)*M13_model_list[61] + 2.*(galaxy_age-10.0)*M13_model_list[62]
        elif galaxy_age >= 11.0 and galaxy_age < 12.0:
            model2 = (12.0-galaxy_age)*M13_model_list[62] + 2.*(galaxy_age-11.0)*M13_model_list[63]
        elif galaxy_age >= 12.0 and galaxy_age < 13.0:
            model2 = (13.0-galaxy_age)*M13_model_list[63] + 2.*(galaxy_age-12.0)*M13_model_list[64]
        elif galaxy_age >= 13.0 and galaxy_age < 14.0:
            model2 = (14.0-galaxy_age)*M13_model_list[64] + 2.*(galaxy_age-13.0)*M13_model_list[65]
        elif galaxy_age >= 14.0 and galaxy_age < 15.0:
            model2 = (15.0-galaxy_age)*M13_model_list[65] + 2.*(galaxy_age-14.0)*M13_model_list[66]
        else:
            model2 = M13_model_list[age_index]



    spectra_extinction = calzetti00(model2[0,:], intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    M13_flux_center = model2[1,:]*spectra_flux_correction
    F_M13_index = 326#126##np.where(abs(model2[0,:]-norm_wavelength)<10.5)[0][0]
    Flux_M13_norm_new = M13_flux_center[F_M13_index]
    smooth_Flux_M13_1Gyr_new = M13_flux_center/Flux_M13_norm_new

    binning_index = find_nearest(model2[0,:],np.median(x))
    if binning_index == 0:
        binning_index = 1
    elif binning_index ==len(x):
        binning_index = len(x)-1
    if (x[int(n/2)]-x[int(n/2)-1]) > (model2[0,binning_index]-model2[0,binning_index-1]):
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(model2[0,binning_index]-model2[0,binning_index-1]))
        model_wave_binned,model_flux_binned = binning_spec_keep_shape(model2[0,:], smooth_Flux_M13_1Gyr_new,binnning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('binning model, model 2', n, (model2[0,binning_index]-model2[0,binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]),binning_size)    
    else:
        binning_size = int((model2[0,binning_index]-model2[0,binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned = binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, model2[0,:], smooth_Flux_M13_1Gyr_new) 
        x2_photo = chisquare_photo(model2[0,:], smooth_Flux_M13_1Gyr_new,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
    
    try: 
        if 0.01<galaxy_age<13 and 0.0<=intrinsic_Av<=4.0 and not np.isinf(0.5*x2+0.5*x2_photo):
            pass
        else:
            x2 = np.inf
            x2_photo = np.inf
    except ValueError: # NaN value case
       x2 = np.inf
       x2_photo = np.inf
       print('ValueError', x2)
    return x2, x2_photo

def minimize_age_AV_vector_weighted_BC03(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(BC03_age_list_num, galaxy_age)
    age_prior = BC03_age_list_num[age_index]
    AV_string = str(intrinsic_Av)
    # print(galaxy_age,age_prior)

    if galaxy_age == age_prior:
        model3_flux = BC03_flux_array[age_index, :7125]
    elif galaxy_age < age_prior:
        age_interval = BC03_age_list_num[age_index+1] - BC03_age_list_num[age_index]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index+1]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index]))*1./age_interval
    elif galaxy_age > age_prior:
        age_interval = BC03_age_list_num[age_index] - BC03_age_list_num[age_index-1]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index-1]))*1./age_interval   

    spectra_extinction = calzetti00(BC03_wave_list_num, intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    BC03_flux_attenuated = model3_flux*spectra_flux_correction
    BC03_flux_norm = BC03_flux_attenuated[2556]
    BC03_flux_attenuated = BC03_flux_attenuated/BC03_flux_norm

    binning_index = find_nearest(BC03_wave_list_num, np.median(x))
    if (x[int(n/2)]-x[int(n/2)-1]) < (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]):
        binning_size = int((BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned = binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, BC03_wave_list_num, BC03_flux_attenuated)
        x2_photo = chisquare_photo(BC03_wave_list_num, BC03_flux_attenuated, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod) 
        # print('bin data', n, binning_size, x2)
    else: 
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]))
        model_wave_binned, model_flux_binned = binning_spec_keep_shape(BC03_wave_list_num, BC03_flux_attenuated,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
        # print('bin model',binning_size, x2)
    # print('binning size, model 3', n, (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)    
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, BC03_wave_list_num, BC03_flux_attenuated)
    # print('BC x2_nu',x2,x2_photo,0.5*weight1*x2+0.5*weight2*x2_photo)
    return 0.5*weight1*x2+0.5*weight2*x2_photo
def lg_minimize_age_AV_vector_weighted_BC03(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(BC03_age_list_num, galaxy_age)
    age_prior = BC03_age_list_num[age_index]
    AV_string = str(intrinsic_Av)

    if galaxy_age == age_prior:
        model3_flux = BC03_flux_array[age_index, :7125]
    elif galaxy_age < age_prior and galaxy_age <1.97500006e+01:
        age_interval = BC03_age_list_num[age_index+1] - BC03_age_list_num[age_index]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index+1]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index]))*1./age_interval
    elif galaxy_age > age_prior and galaxy_age <1.97500006e+01:
        age_interval = BC03_age_list_num[age_index] - BC03_age_list_num[age_index-1]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index-1]))*1./age_interval   
    else:
        model3_flux = BC03_flux_array[-1, :7125]   
    
    spectra_extinction = calzetti00(BC03_wave_list_num, intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    BC03_flux_attenuated = model3_flux*spectra_flux_correction
    BC03_flux_norm = BC03_flux_attenuated[2556]
    BC03_flux_attenuated = BC03_flux_attenuated/BC03_flux_norm

    binning_index = find_nearest(BC03_wave_list_num, np.median(x))
    if (x[int(n/2)]-x[int(n/2)-1]) < (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]):
        binning_size = int((BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, BC03_wave_list_num, BC03_flux_attenuated) 
        x2_photo = chisquare_photo(BC03_wave_list_num, BC03_flux_attenuated, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod) 

        # print('bin data', binning_size, x2)
    else: 
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]))
        model_wave_binned, model_flux_binned = binning_spec_keep_shape(BC03_wave_list_num, BC03_flux_attenuated,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned)
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)
 
        # print('bin model',binning_size, x2)
    # print('binning size, model 3', n, (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)    
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, BC03_wave_list_num, BC03_flux_attenuated)
    
    if 0.01<galaxy_age<13 and 0.0<intrinsic_Av<4.0 and not np.isinf(0.5*x2+0.5*1e-3*x2_photo):
        return np.log(np.exp(-0.5*(0.5*weight1*x2+0.5*weight2*x2_photo)))
    else:
        return -np.inf
def minimize_age_AV_vector_weighted_BC03_mod_no_weight_return_flux(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(BC03_age_list_num, galaxy_age)
    age_prior = BC03_age_list_num[age_index]
    AV_string = str(intrinsic_Av)

    if galaxy_age == age_prior:
        model3_flux = BC03_flux_array[age_index, :7125]
    elif galaxy_age < age_prior and galaxy_age <1.97500006e+01:
        age_interval = BC03_age_list_num[age_index] - BC03_age_list_num[age_index-1]
        model3_flux = (BC03_flux_array[age_index-1, :7125]*(BC03_age_list_num[age_index]-galaxy_age)\
                            + BC03_flux_array[age_index, :7125]*(galaxy_age-BC03_age_list_num[age_index-1]))*1./age_interval
    elif galaxy_age > age_prior and galaxy_age <1.97500006e+01:
        age_interval = BC03_age_list_num[age_index+1] - BC03_age_list_num[age_index]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index+1]-galaxy_age)\
                            + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index]))*1./age_interval   

    spectra_extinction = calzetti00(BC03_wave_list_num, intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    BC03_flux_attenuated = model3_flux*spectra_flux_correction
    BC03_flux_norm = BC03_flux_attenuated[2556]
    BC03_flux_attenuated = BC03_flux_attenuated/BC03_flux_norm

    binning_index = find_nearest(BC03_wave_list_num, np.median(x))
    if (x[int(n/2)]-x[int(n/2)-1]) < (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]):
        binning_size = int((BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, BC03_wave_list_num, BC03_flux_attenuated) 
        x2_photo = chisquare_photo(BC03_wave_list_num, BC03_flux_attenuated, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod) 
        # print('bin data', binning_size, x2)
    else: 
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]))
        model_wave_binned, model_flux_binned = binning_spec_keep_shape(BC03_wave_list_num, BC03_flux_attenuated,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # print('bin model',binning_size, x2)
    # print('binning size, model 3', n, (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)     
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, BC03_wave_list_num, BC03_flux_attenuated)
    
    return 0.5*weight1*x2+0.5*weight2*x2_photo,BC03_flux_attenuated
def minimize_age_AV_vector_weighted_BC03_return_chi2_sep(X):
    galaxy_age= X[0]
    intrinsic_Av = X[1]
    n=len(x)
    age_index = find_nearest(BC03_age_list_num, galaxy_age)
    age_prior = BC03_age_list_num[age_index]
    AV_string = str(intrinsic_Av)

    if galaxy_age == age_prior:
        model3_flux = BC03_flux_array[age_index, :7125]
    elif galaxy_age < age_prior:
        age_interval = BC03_age_list_num[age_index+1] - BC03_age_list_num[age_index]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index+1]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index]))*1./age_interval
    elif galaxy_age > age_prior:
        age_interval = BC03_age_list_num[age_index] - BC03_age_list_num[age_index-1]
        model3_flux = (BC03_flux_array[age_index, :7125]*(BC03_age_list_num[age_index]-galaxy_age) + BC03_flux_array[age_index+1, :7125]*(galaxy_age-BC03_age_list_num[age_index-1]))*1./age_interval   

    spectra_extinction = calzetti00(BC03_wave_list_num, intrinsic_Av, 4.05)
    spectra_flux_correction = 10**(-0.4*spectra_extinction)
    BC03_flux_attenuated = model3_flux*spectra_flux_correction
    BC03_flux_norm = BC03_flux_attenuated[2556]
    BC03_flux_attenuated = BC03_flux_attenuated/BC03_flux_norm

    binning_index = find_nearest(BC03_wave_list_num, np.median(x))
    if (x[int(n/2)]-x[int(n/2)-1]) < (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]):
        binning_size = int((BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1])/(x[int(n/2)]-x[int(n/2)-1]))
        x_binned,y_binned,y_err_binned=binning_spec_keep_shape_x(x,y,y_err,binning_size)
        x2 = reduced_chi_square(x_binned, y_binned, y_err_binned, BC03_wave_list_num, BC03_flux_attenuated) 
        x2_photo = chisquare_photo(BC03_wave_list_num, BC03_flux_attenuated, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod) 
        # print('bin data', binning_size, x2)
    else: 
        binning_size = int((x[int(n/2)]-x[int(n/2)-1])/(BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]))
        model_wave_binned, model_flux_binned = binning_spec_keep_shape(BC03_wave_list_num, BC03_flux_attenuated,binning_size)
        x2 = reduced_chi_square(x, y, y_err, model_wave_binned, model_flux_binned) 
        x2_photo = chisquare_photo(model_wave_binned, model_flux_binned,redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod)

        # print('bin model',binning_size, x2)
    # print('binning size, model 3', n, (BC03_wave_list_num[binning_index]-BC03_wave_list_num[binning_index-1]), (x[int(n/2)]-x[int(n/2)-1]), binning_size)    
    # x2_photo = reduced_chi_square(wave_list, photometric_flux, photometric_flux_err, BC03_wave_list_num, BC03_flux_attenuated)
    
    return x2,x2_photo

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    # print('find nearest idx searchsorted:', idx)
    if np.isnan(idx):
        print('find nearest',idx,value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1#array[idx-1]
    else:
        return idx#array[idx]
def all_same(items):
    return all(x == items[0] for x in items)
def reduced_chi_square(data_wave,data,data_err,model_wave,model):
    n=len(data_wave)
    chi_square = 0
    for i in range(n):
        model_flux_interp = np.interp(data_wave[i], model_wave, model)
        chi_square += (data[i]-model_flux_interp)**2/(data_err[i]**2)
    # print('spectra chisquare processes new',i,chi_square, data_wave[i],model_flux_interp)
    dof = n-2
    reduced_chi_square = chi_square/dof
    return reduced_chi_square
def chisquare_photo(model_wave, model_flux, redshift_1,wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod):
    """
    work in the observed frame
    """
    tik = time.clock()
    model_wave = model_wave*(1+redshift_1)
    model_flux = model_flux

    filter_array_index= np.arange(1,37)

    #    SNR Mask
    mask_SNR3_photo = np.where(photometric_flux/photometric_flux_err>3.)
    photometric_flux = photometric_flux[mask_SNR3_photo]
    photometric_flux_err = photometric_flux_err[mask_SNR3_photo]
    photometric_flux_err_mod = photometric_flux_err_mod[mask_SNR3_photo]
    filter_array_index = filter_array_index[mask_SNR3_photo]

    photometry_list = np.zeros(len(photometric_flux))
    photometry_list_index = 0
    # print('masked filter array index:',filter_array_index)
    
    for i in filter_array_index:

        sum_flambda_AB_K = 0
        sum_transmission = 0
        length = 0
        filter_curve = filter_curve_list[i-1]

        wave_inter = np.zeros(len(model_wave))
        wave_inter[:-1] = np.diff(model_wave)
        index = np.where(model_wave<filter_curve[-1,0])[0]#[0]
        wave = model_wave[index]
        flux = model_flux[index]
        wave_inter = wave_inter[index]
        index = np.where(wave>filter_curve[0,0])
        wave = wave[index]
        flux = flux[index]
        wave_inter = wave_inter[index]
        transmission = np.interp(wave, filter_curve[:,0], filter_curve[:,1])

        n = len(flux)
        if n!= 0 and n!=1:
            for j in range(n):
                try:
                    if all_same(wave_inter):
                        flambda_AB_K = flux[j]*transmission[j]
                        sum_flambda_AB_K += flambda_AB_K
                        sum_transmission += transmission[j]
                        length = length+1
                    else:
                        flambda_AB_K = flux[j]*transmission[j]*wave_inter[j]
                        sum_flambda_AB_K += flambda_AB_K
                        sum_transmission += transmission[j]*wave_inter[j]
                        length = length+1
                except:
                    print('Error',n,transmission_index, j,wave[j],filter_curve[0,0],filter_curve[-1,0])
                     
        elif n==1:
            flambda_AB_K = flux[0]*transmission[0]
            sum_flambda_AB_K += flambda_AB_K*wave_inter
            sum_transmission += np.sum(transmission)*wave_inter
            length = length+1
        
        if length == 0:
            photometry_list[photometry_list_index]=0
        else:
            photometry_list[photometry_list_index] = sum_flambda_AB_K/sum_transmission
        photometry_list_index += 1

    chisquare_photo_list = ((photometric_flux-photometry_list)/photometric_flux_err_mod)**2
    
    tok = time.clock()
    dof = len(chisquare_photo_list)-2
    reduced_chi_square_photo = np.sum(chisquare_photo_list)/dof

    return reduced_chi_square_photo



nsteps=3000
current_dir = '/home/siqi/TAPS/TAPS/'
outcome_dir = 'outcome/'
date = '20200329'
plot_dir = 'plot/'+str(date)+'_goodss/'


## Prepare the filter curves
tik = time.time()
filter_fn_list = []
filter_curve_list=[]
filter_curve_fit_list=[]
path = "/home/siqi/TAPS/TAPS/filter/goodss/"
import glob, os
os.chdir(path)
for i in range(1,37):
    for file in glob.glob("f"+str(i)+"_*"):
        print(file)
        fn = path+file
        filter_fn_list.append(fn)
    filter_curve = np.loadtxt(fn)
    filter_curve_list.append(filter_curve)
    filter_f = interpolate.interp1d(filter_curve[:,0], filter_curve[:,1])
    filter_curve_fit_list.append(filter_f)
tok = time.time()
print('Time reading the filter curves and without generate filter functions:',tok-tik)

for i in range(len(df)):
    row = i
    
    [ID, OneD_1, redshift_1, mag_1] = read_spectra(row)
    print(i, ID)        
    ID_no = ID-1
    redshift = df_photometry.loc[ID_no].z_spec

    region = df.region[row]
    intrinsic_Av = df_fast.loc[ID-1].Av
    print('intrinsic Av:'+str(intrinsic_Av))
    galaxy_age = 10**(df_fast.loc[ID-1].lage)/1e9
    print('Galaxy age:', galaxy_age)
    A_v=0.0207    
    c=3e10
    
    chi_square_list.loc[row,'ID'] = float(ID)
    chi_square_list.loc[row,'region'] = region
    chi_square_list.loc[row,'field'] = 'goodss'

# Photometry
    # ESO GOODS  |  Nonino et al. 2009
    # Paranal_VIMOS U and VIMOS R
    U_wave = 3722
    U_band = 375.5/2.
    U = df_photometry.loc[ID_no].f_U/((U_wave)**2)*c*1e8*3.63e-30
    U_err = df_photometry.loc[ID_no].e_U/((U_wave)**2)*c*1e8*3.63e-30

    R_wave = 6449.7
    R_band = 1286.3/2.
    R = df_photometry.loc[ID_no].f_R/((R_wave)**2)*c*1e8*3.63e-30
    R_err = df_photometry.loc[ID_no].e_R/((R_wave)**2)*c*1e8*3.63e-30
    
    
    #  GaBoDs     | Hildebrandt et al. 2006, Erben 2005      
    # LaSilla WFI ESO 841-845.dat
    U38_wave = 3706 # ESO 841
    U38_band = 357./2.
    U38 = df_photometry.loc[ID_no].f_U38/((U38_wave)**2)*c*1e8*3.63e-30
    U38_err = df_photometry.loc[ID_no].e_U38/((U38_wave)**2)*c*1e8*3.63e-30

    B_wave = 4554 # 842
    B_band = 915./2.
    B = df_photometry.loc[ID_no].f_B/((B_wave)**2)*c*1e8*3.63e-30
    B_err = df_photometry.loc[ID_no].e_B/((B_wave)**2)*c*1e8*3.63e-30

    V_wave = 5343 #843
    V_band = 900./2.
    V = df_photometry.loc[ID_no].f_V/((V_wave)**2)*c*1e8*3.63e-30
    V_err = df_photometry.loc[ID_no].e_V/((V_wave)**2)*c*1e8*3.63e-30

    Rc_wave = 6411 #844
    Rc_band = 1602./2.
    Rc = df_photometry.loc[ID_no].f_Rc/((Rc_wave)**2)*c*1e8*3.63e-30
    Rc_err = df_photometry.loc[ID_no].e_Rc/((Rc_wave)**2)*c*1e8*3.63e-30

    I_wave = 8554 # 845
    I_band = 1504./2.
    I = df_photometry.loc[ID_no].f_I/((I_wave)**2)*c*1e8*3.63e-30
    I_err = df_photometry.loc[ID_no].e_I/((I_wave)**2)*c*1e8*3.63e-30
    
    
    # MUSYC      | Cardamone et al. 2010                 
    #f_IA427 e_IA427 f_IA445 e_IA445 f_IA505 e_IA505 f_IA527 e_IA527 f_IA550 e_IA550 f_IA574 
    #e_IA574 f_IA598 e_IA598 f_IA624 e_IA624 f_IA651 e_IA651 f_IA679 e_IA679 f_IA738 e_IA738 
    #f_IA767 e_IA767 f_IA797 e_IA797 f_IA856 e_IA856 
    # IA427: 4253, IA445: 4445, IA464: 4631, IA484: 4843, IA505: 5059, IA527: 5256, IA550: 5492, IA574: 5760
    # IA598: 6003, IA624: 6227, IA651: 6491, IA679: 6778, IA709: 7070, IA738: 7356, IA768: 7676, IA797: 7962
    # IA827:8243, IA856:8562
    IA427_wave = 4253
    IA427_band = 210./2.
    IA427 = df_photometry.loc[ID_no].f_IA427/IA427_wave**2*c*1e8*3.63e-30
    IA427_err = df_photometry.loc[ID_no].e_IA427/IA427_wave**2*c*1e8*3.63e-30
    
    IA445_wave = 4445
    IA445_band = 204./2.
    IA445 = df_photometry.loc[ID_no].f_IA445/IA445_wave**2*c*1e8*3.63e-30
    IA445_err = df_photometry.loc[ID_no].e_IA427/IA445_wave**2*c*1e8*3.63e-30
    
    IA505_wave = 5059
    IA505_band = 234./2.
    IA505 = df_photometry.loc[ID_no].f_IA505/IA505_wave**2*c*1e8*3.63e-30
    IA505_err = df_photometry.loc[ID_no].e_IA505/IA505_wave**2*c*1e8*3.63e-30
    
    IA527_wave = 5256
    IA527_band = 243./2.
    IA527 = df_photometry.loc[ID_no].f_IA527/IA527_wave**2*c*1e8*3.63e-30
    IA527_err = df_photometry.loc[ID_no].e_IA527/IA527_wave**2*c*1e8*3.63e-30
    
    IA550_wave = 5492
    IA550_band = 276./2.
    IA550 = df_photometry.loc[ID_no].f_IA550/IA550_wave**2*c*1e8*3.63e-30
    IA550_err = df_photometry.loc[ID_no].e_IA550/IA550_wave**2*c*1e8*3.63e-30
    
    IA574_wave = 5760
    IA574_band = 276./2.
    IA574 = df_photometry.loc[ID_no].f_IA574/IA574_wave**2*c*1e8*3.63e-30
    IA574_err = df_photometry.loc[ID_no].e_IA574/IA574_wave**2*c*1e8*3.63e-30
    
    IA598_wave = 6003
    IA598_band = 297./2.
    IA598 = df_photometry.loc[ID_no].f_IA598/IA598_wave**2*c*1e8*3.63e-30
    IA598_err = df_photometry.loc[ID_no].e_IA598/IA598_wave**2*c*1e8*3.63e-30
    
    IA624_wave = 6227
    IA624_band = 300./2.
    IA624 = df_photometry.loc[ID_no].f_IA624/IA624_wave**2*c*1e8*3.63e-30
    IA624_err = df_photometry.loc[ID_no].e_IA624/IA624_wave**2*c*1e8*3.63e-30
    
    IA651_wave = 6491
    IA651_band = 324./2.
    IA651 = df_photometry.loc[ID_no].f_IA651/IA651_wave**2*c*1e8*3.63e-30
    IA651_err = df_photometry.loc[ID_no].e_IA651/IA651_wave**2*c*1e8*3.63e-30
    
    IA679_wave = 6778
    IA679_band = 339./2.
    IA679 = df_photometry.loc[ID_no].f_IA679/IA679_wave**2*c*1e8*3.63e-30
    IA679_err = df_photometry.loc[ID_no].e_IA679/IA679_wave**2*c*1e8*3.63e-30
    
    IA738_wave = 7356
    IA738_band = 324./2.
    IA738 = df_photometry.loc[ID_no].f_IA738/IA738_wave**2*c*1e8*3.63e-30
    IA738_err = df_photometry.loc[ID_no].e_IA738/IA738_wave**2*c*1e8*3.63e-30
    
    IA767_wave = 7676
    IA767_band = 366./2.
    IA767 = df_photometry.loc[ID_no].f_IA767/IA767_wave**2*c*1e8*3.63e-30
    IA767_err = df_photometry.loc[ID_no].e_IA767/IA767_wave**2*c*1e8*3.63e-30
    
    IA797_wave = 7962
    IA797_band = 354./2.
    IA797 = df_photometry.loc[ID_no].f_IA797/IA797_wave**2*c*1e8*3.63e-30
    IA797_err = df_photometry.loc[ID_no].e_IA797/IA797_wave**2*c*1e8*3.63e-30
    
    IA856_wave = 8562
    IA856_band = 324./2.
    IA856 = df_photometry.loc[ID_no].f_IA856/IA856_wave**2*c*1e8*3.63e-30
    IA856_err = df_photometry.loc[ID_no].e_IA856/IA856_wave**2*c*1e8*3.63e-30
     
    # GOODS      |Giavalisco et al. 2004 | F435W,  F606W, F775W, F850LP|
    # https://uknowledge.uky.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1264&context=physastron_facpub
    # Also on Guo et al. 2013
    F435W_wave = 4317
    F435W_band = 920./2.
    F435W = df_photometry.loc[ID_no].f_F435W/((F435W_wave)**2)*c*1e8*3.63e-30
    F435W_err = df_photometry.loc[ID_no].e_F435W/((F435W_wave)**2)*c*1e8*3.63e-30
    
    F606W_wave = 5918
    F606W_band = 2324./2.
    F606W = df_photometry.loc[ID_no].f_F606W/((F606W_wave)**2)*c*1e8*3.63e-30
    F606W_err = df_photometry.loc[ID_no].e_F606W/((F606W_wave)**2)*c*1e8*3.63e-30
    
    F775W_wave = 7693
    F775W_band = 1511./2.
    F775W = df_photometry.loc[ID_no].f_F775W/((F775W_wave)**2)*c*1e8*3.63e-30
    F775W_err = df_photometry.loc[ID_no].e_F775W/((F775W_wave)**2)*c*1e8*3.63e-30
    
    F850LP_wave = 9055
    F850LP_band = 1236./2.
    F850LP = df_photometry.loc[ID_no].f_F850LP/((F850LP_wave)**2)*c*1e8*3.63e-30
    F850LP_err = df_photometry.loc[ID_no].e_F850LP/((F850LP_wave)**2)*c*1e8*3.63e-30
    
    
    # CANDELS    | Koekemoer et al. 2011, what wavelength this should take? : the same as above        
    F606Wcand_wave = 5918
    F606Wcand_band = 2324./2.
    F606Wcand = df_photometry.loc[ID_no].f_F606Wcand/((F606Wcand_wave)**2)*c*1e8*3.63e-30
    F606Wcand_err = df_photometry.loc[ID_no].e_F606Wcand/((F606Wcand_wave)**2)*c*1e8*3.63e-30
    
    F814Wcand_wave = 8047
    F814Wcand_band = 1826./2.
    F814Wcand = df_photometry.loc[ID_no].f_F814Wcand/((F814Wcand_wave)**2)*c*1e8*3.63e-30
    F814Wcand_err = df_photometry.loc[ID_no].e_F814Wcand/((F814Wcand_wave)**2)*c*1e8*3.63e-30
    
    F850LPcand_wave = 9055
    F850LPcand_band = 1236./2.
    F850LPcand = df_photometry.loc[ID_no].f_F850LPcand/((F850LPcand_wave)**2)*c*1e8*3.63e-30
    F850LPcand_err = df_photometry.loc[ID_no].e_F850LPcand/((F850LPcand_wave)**2)*c*1e8*3.63e-30
        
    # CANDELS    | Grogin et al. 2011, Koekemoer et al. 2011|
    F125W_wave = 12486
    F125W_band = 3005./2.
    F125W = df_photometry.loc[ID_no].f_F125W/((F125W_wave)**2)*c*1e8*3.63e-30
    F125W_err = df_photometry.loc[ID_no].e_F125W/((F125W_wave)**2)*c*1e8*3.63e-30
    
    F160W_wave = 15370
    F160W_band = 2874./2.
    F160W = df_photometry.loc[ID_no].f_F160W/((F160W_wave)**2)*c*1e8*3.63e-30 #http://www.stsci.edu/hst/wfc3/design/documents/handbooks/currentIHB/c07_ir06.html
    F160W_err = df_photometry.loc[ID_no].e_F160W/((F160W_wave)**2)*c*1e8*3.63e-30
    
    # 3D-HST     | Brammer et al. 2012        
    F140W_wave = 13635
    F140W_band = 3947./2.
    F140W = df_photometry.loc[ID_no].f_F140W/((F140W_wave)**2)*c*1e8*3.63e-30 #http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=HST/WFC3_IR.F140W
    F140W_err = df_photometry.loc[ID_no].e_F140W/((F140W_wave)**2)*c*1e8*3.63e-30

    #  J, H, Ks    | ESO/GOODS  | Retzlaff et al. 2010, Wuyts et al. 2008  |
    #  J: 1.25, H: 1.65, Ks: 2.1605
    # ISSAC J, H, Ks: FJ/FJ/FK_BB.ASCII
    J_wave = 1.25e4
    J_band = 0.29e4/2.
    J = df_photometry.loc[ID_no].f_J/J_wave**2*c*1e8*3.63e-30
    J_err = df_photometry.loc[ID_no].e_J/J_wave**2*c*1e8*3.63e-30
    
    H_wave = 1.65e4
    H_band = 0.3e4/2.
    H = df_photometry.loc[ID_no].f_H/H_wave**2*c*1e8*3.63e-30
    H_err = df_photometry.loc[ID_no].e_H/H_wave**2*c*1e8*3.63e-30
    
    Ks_wave = 2.1605e4
    Ks_band = 0.27e4/2.
    Ks = df_photometry.loc[ID_no].f_Ks/Ks_wave**2*c*1e8*3.63e-30
    Ks_err = df_photometry.loc[ID_no].e_Ks/Ks_wave**2*c*1e8*3.63e-30
    
    # J, Ks    | TENIS      | Hsieh et al. 2012       
    # J: 12481, Ks: 21338 tenisJ
    # WIRCam J and Ks
    tenisJ_wave = 12481
    tenisJ_band = 1588./2.
    tenisJ = df_photometry.loc[ID_no].f_tenisJ/tenisJ_wave**2*c*1e8*3.63e-30
    tenisJ_err = df_photometry.loc[ID_no].e_tenisJ/tenisJ_wave**2*c*1e8*3.63e-30
    
    tenisK_wave = 21338
    tenisK_band = 3270./2.
    tenisK = df_photometry.loc[ID_no].f_tenisK/tenisK_wave**2*c*1e8*3.63e-30
    tenisK_err = df_photometry.loc[ID_no].e_tenisK/tenisK_wave**2*c*1e8*3.63e-30
    
    wave_list = np.array([U_wave, R_wave, U38_wave, B_wave, V_wave, Rc_wave, I_wave, \
                        IA427_wave, IA445_wave, IA505_wave, IA527_wave, IA550_wave,\
                        IA574_wave, IA598_wave, IA624_wave, IA651_wave, IA679_wave, IA738_wave, IA767_wave, IA797_wave, IA856_wave,\
                        F435W_wave, F606W_wave, F775W_wave, F850LP_wave, F606Wcand_wave, F814Wcand_wave, F850LPcand_wave,\
                        F125W_wave, F140W_wave, F160W_wave, J_wave, H_wave, Ks_wave, tenisJ_wave, tenisK_wave])

    band_list = np.array([U_band, R_band, U38_band, B_band, V_band, Rc_band, I_band, \
                        IA427_band, IA445_band, IA505_band, IA527_band, IA550_band,\
                        IA574_band, IA598_band, IA624_band, IA651_band, IA679_band, IA738_band, IA767_band, IA797_band, IA856_band,\
                        F435W_band, F606W_band, F775W_band, F850LP_band, F606Wcand_band, F814Wcand_band, F850LPcand_band,\
                        F125W_band, F140W_band, F160W_band, J_band, H_band, Ks_band, tenisJ_band, tenisK_band])
    
    photometric_flux = np.array([U, R, U38, B, V, Rc, I, IA427, IA445, IA505, IA527, IA550, IA574, IA598, IA624, IA651, IA679, IA738, IA767, IA797, IA856, \
                                F435W, F606W, F775W, F850LP, F606Wcand, F814Wcand, F850LPcand, F125W, F140W, F160W, J, H, Ks, tenisJ, tenisK])
    photometric_flux_err = np.array([U_err, R_err, U38_err, B_err, V_err, Rc_err,\
                                     I_err, IA427_err, IA445_err, IA505_err, IA527_err, IA550_err, IA574_err, IA598_err, IA624_err, IA651_err, IA679_err,\
                                     IA738_err, IA767_err, IA797_err, IA856_err, \
                                     F435W_err, F606W_err, F775W_err, F850LP_err, F606Wcand_err, F814Wcand_err, F850LPcand_err,\
                                     F125W_err, F140W_err, F160W_err, J_err, H_err, Ks_err, tenisJ_err, tenisK_err])
    
    photometric_flux_err_mod = np.array([U_err+0.1*U, R_err+0.1*R, U38_err+0.1*U38, B_err+0.1*B, V_err+0.1*V, Rc_err+0.1*Rc, I_err+0.1*I,\
                                    IA427_err+0.1*IA427, IA445_err+0.1*IA445, IA505_err+0.1*IA505, IA527_err+0.1*IA527, IA550_err+0.1*IA550, IA574_err+0.1*IA574,\
                                    IA598_err+0.1*IA598, IA624_err+0.1*IA624, IA651_err+0.1*IA651, IA679_err+0.1*IA679, IA738_err+0.1*IA738, IA767_err+0.1*IA767,\
                                    IA797_err+0.1*IA797, IA856_err+0.1*IA856, \
                                    F435W_err+0.03*F435W,  F606W_err+0.03*F606W, F775W_err+0.03*F775W, F850LP_err+0.03*F850LP,\
                                    F606Wcand_err+0.03*F606Wcand, F814Wcand_err+0.03*F814Wcand, F850LPcand_err+0.03*F850LPcand, F125W_err+0.03*F125W, F140W_err+0.03*F140W, F160W_err+0.03*F160W,\
                                    J_err+0.1*J, H_err+0.1*H, Ks_err+0.1*Ks, tenisJ_err+0.1*tenisJ, tenisK_err+0.1*tenisK])
#-------------------------------------------------Initial Reduce the spectra ----------------------------------------------------------
    print('-------------------------------------Initial fit ---------------------------------------------------------------------------------------')
    [x, y, y_err, wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod] = \
    derive_1D_spectra_Av_corrected(OneD_1, redshift_1, row, wave_list, band_list, photometric_flux, photometric_flux_err, photometric_flux_err_mod, A_v)
    if redshift< 0.49:
        try:
            chi_square_list.loc[row,'grism_index'] = Lick_index_ratio(x,y)
        except:
            pass
    # print(y)
    # Testing fitting a line
    photo_list_for_scaling = []
    photo_err_list_for_scaling = []
    grism_flux_list_for_scaling = []
    grism_flux_err_list_for_scaling = []
    grism_wave_list_for_scaling =[]
    for i in range(len(wave_list)):
        if wave_list[i]-band_list[i] > x[0] and wave_list[i] + band_list[i] < x[-1]:
            scale_index = find_nearest(x, wave_list[i])
            photo_list_for_scaling.append(photometric_flux[i])
            photo_err_list_for_scaling.append(photometric_flux_err[i])
            grism_flux_list_for_scaling.append(y[scale_index])
            grism_flux_err_list_for_scaling.append(y_err[scale_index])
            grism_wave_list_for_scaling.append(x[scale_index])
    photo_array_for_scaling = np.array(photo_list_for_scaling)
    photo_err_array_for_scaling = np.array(photo_err_list_for_scaling)
    grism_flux_array_for_scaling = np.array(grism_flux_list_for_scaling)
    grism_flux_err_array_for_scaling = np.array(grism_flux_err_list_for_scaling)
    grism_wave_array_for_scaling = np.array(grism_wave_list_for_scaling)
    print('Number of photometric points for rescaling:',len(photo_array_for_scaling))
    print(np.mean(photo_array_for_scaling/grism_flux_array_for_scaling))
    
    rescaling_err = 1/np.sqrt(1./grism_flux_array_for_scaling*photo_err_array_for_scaling**2
                       + photo_array_for_scaling/grism_flux_array_for_scaling**2*grism_flux_err_array_for_scaling**2)
    ## 0th order
    number_of_poly = 0#np.floor_divide(len(photo_array_for_scaling),3)-1
    p= np.polyfit(grism_wave_list_for_scaling,\
                           photo_array_for_scaling/grism_flux_array_for_scaling, number_of_poly,\
                           w=rescaling_err)
    y_fit = np.polyval(p,x)
    y = y_fit*y

    print('photo flux: ',photometric_flux,len(photometric_flux[photometric_flux>0]))
# Using bounds to constrain
# Test with M05 models
    print('____________________M05_________________________ Optimization__________________________')
    X = np.array([galaxy_age, intrinsic_Av])
    bnds = ((0.01, 13.0), (0.0, 4.0))
    sol = optimize.minimize(minimize_age_AV_vector_weighted, X, bounds = bnds, method='SLSQP')#, options = {'disp': True})
    print('Optimized weighted reduced chisqure result:', sol)
    [age_prior_optimized, AV_prior_optimized] = sol.x
    X = sol.x
    x2_optimized = minimize_age_AV_vector_weighted(X)
    x2_spec, x2_phot = minimize_age_AV_vector_weighted_return_chi2_sep(X)
    chi_square_list.loc[row,'M05_age_opt'] = X[0]#"{0:.2f}".format(X[0])
    chi_square_list.loc[row,'M05_AV_opt'] = X[1]#"{0:.2f}".format(X[1])
    chi_square_list.loc[row,'x2_M05_opt'] = x2_optimized
    chi_square_list.loc[row,'x2_spectra_M05_opt'] = x2_spec
    chi_square_list.loc[row,'x2_photo_M05_opt'] = x2_phot

    #--- Plot
    X=sol.x
    n = len(x)
    fig1 = plt.figure(figsize=(20,10))
    frame1 = fig1.add_axes((.1,.35,.8,.6))
    plt.step(x, y, color='r',lw=3)
    plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
    plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
    model_wave,model_flux =minimize_age_AV_vector_weighted_return_flux(X)[1:]
    plt.plot(model_wave, model_flux, color='k',label='TP-AGB heavy',lw=0.5)
    plt.xlim([2.5e3,1.9e4])
    plt.ylim([0.05, 1.2])
    plt.semilogx()
    plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(loc='upper right',fontsize=24)
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)

    frame2 = fig1.add_axes((.1,.2,.8,.15))  
    relative_spectra = np.zeros([1,n])
    relative_spectra_err = np.zeros([1,n])
    relative_sigma = np.zeros([1,n])
    index0 = 0
    for wave in x:
        if y[index0]>0.25 and y[index0]<1.35:
            index = find_nearest(model_wave, wave);#print index
            relative_spectra[0, index0] = y[index0]/model_flux[index]
            relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
            relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
            index0 = index0+1
    plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
    index0 = 0
    for i in range(len(wave_list)):
        try:
            index = find_nearest(model_wave, wave_list[i])
        except:
            pass
        plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
        index0 = index0+1
    plt.xlim([2.5e3,1.9e4])
    plt.semilogx()
    plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
    plt.ylim([-5,5])
    plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
    plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    figname=current_dir+outcome_dir+plot_dir+'GOODSS_M05_SSP_opt_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
    plt.savefig(figname)
    plt.clf()

    with Pool() as pool:
        ndim, nwalkers = 2, 10
        tik = time.clock()
        p0 = [sol.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)
        samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
        print(np.size(samples))
        samples = samples[(samples[:,0] > age_prior_optimized*0.1) & (samples[:,0] < age_prior_optimized*2.0) & (samples[:,1] < AV_prior_optimized*3.0)]
        tok = time.clock()
        multi_time = tok-tik
        print(np.size(samples))
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        print('Time to run M05 MCMC:'+str(tok-tik)) 
        print('sample size:',samples.size)       
    try:
        if samples.size > 1e3:
            value2 = np.percentile(samples, 50, axis=0)
            X = np.percentile(samples, 50, axis=0)
            [std_age_prior_optimized, std_AV_prior_optimized] = np.std(samples, axis=0)
            plt.figure(figsize=(32,32),dpi=100)
            fig = corner.corner(samples,
                 labels=["age(Gyr)", r"$\rm A_V$"],
                 truths=[age_prior_optimized, AV_prior_optimized],
                 levels = (1-np.exp(-0.5),),
                 show_titles=True,title_kwargs={'fontsize':12},
                                quantiles=(0.16,0.5, 0.84))
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(X[i], color="g")
                ax.axvline(value2[i],color='r')
            # Loop over the histograms
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(X[xi], color="g")
                    ax.axvline(value2[xi], color="r")
                    ax.axhline(X[yi], color="g")
                    ax.axhline(value2[yi], color="r")
                    ax.plot(X[xi], X[yi], "sg")
                    ax.plot(value2[xi],value2[yi],'sr')
            plt.rcParams.update({'font.size': 12})
            figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_M05_"+str(nsteps)+'_'+str(ID)+'_'+str(region)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
            fig.savefig(figname)
            fig.clf()
            print('MCMC results maximum Likelihood Point M05:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))

            #--- Plot
            X = np.percentile(samples, 50, axis=0)
            x2_optimized = minimize_age_AV_vector_weighted(X)
            x2_spec, x2_phot = minimize_age_AV_vector_weighted_return_chi2_sep(X)
            chi_square_list.loc[row,'M05_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
            chi_square_list.loc[row,'M05_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
            chi_square_list.loc[row,'x2_M05_MCMC50'] = x2_optimized
            chi_square_list.loc[row,'x2_spectra_M05_MCMC50'] = x2_spec
            chi_square_list.loc[row,'x2_photo_M05_MCMC50'] = x2_phot
            chi_square_list.loc[row,'M05_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
            chi_square_list.loc[row,'M05_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])

            n = len(x)
            fig1 = plt.figure(figsize=(20,10))
            frame1 = fig1.add_axes((.1,.35,.8,.6))
            plt.step(x, y, color='r',lw=3)
            plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
            plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
            model_wave,model_flux =minimize_age_AV_vector_weighted_return_flux(X)[1:]
            plt.plot(model_wave, model_flux, color='k',label='TP-AGB heavy',lw=0.5)
            plt.xlim([2.5e3,1.9e4])
            plt.ylim([0.05, 1.2])#plt.ylim([ymin,ymax])
            plt.semilogx()
            plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.legend(loc='upper right',fontsize=24)
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
            
            frame2 = fig1.add_axes((.1,.2,.8,.15))  
            relative_spectra = np.zeros([1,n])
            relative_spectra_err = np.zeros([1,n])
            relative_sigma = np.zeros([1,n])
            index0 = 0
            for wave in x:
                if y[index0]>0.25 and y[index0]<1.35:
                    index = find_nearest(model_wave, wave);#print index
                    relative_spectra[0, index0] = y[index0]/model_flux[index]
                    relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                    relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                    index0 = index0+1
            plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
            index0 = 0
            for i in range(len(wave_list)):
                try:
                    index = find_nearest(model_wave, wave_list[i])
                except:
                    pass
                plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
                index0 = index0+1
            plt.xlim([2.5e3,1.9e4])
            plt.semilogx()
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)            
            plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
            plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
            plt.ylim([-5,5])
            plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
            plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
            figname=current_dir+outcome_dir+plot_dir+'GOODSS_M05_SSP_MCMC50_'+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
            plt.savefig(figname)
            plt.clf()
        else :
            with Pool() as pool:
                print('modified steps:',nsteps)
                ndim, nwalkers = 2, 10
                tik = time.clock()
                p0 = [sol.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted, pool=pool)
                sampler.run_mcmc(p0, nsteps*2, progress=True)
                samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
                samples = samples[(samples[:,0] > age_prior_optimized*0.1) & (samples[:,0] < age_prior_optimized*2.0) & (samples[:,1] < AV_prior_optimized*3.0)]
                tok = time.clock()
                multi_time = tok-tik
                print('modified sample size',np.size(samples))
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                print('Time to run M05 MCMC:'+str(tok-tik))     
            if samples.size > 1e3:
                value2 = np.percentile(samples, 50, axis=0)
                X = np.percentile(samples,50,axis=0)
                [std_age_prior_optimized, std_AV_prior_optimized] = np.std(samples, axis=0)
                plt.figure(figsize=(32,32),dpi=100)
                fig = corner.corner(samples,
                     labels=["age(Gyr)", r"$\rm A_V$"],
                     truths=[age_prior_optimized, AV_prior_optimized],
                     levels = (1-np.exp(-0.5),),
                     show_titles=True,title_kwargs={'fontsize':12},
                                    quantiles=(0.16,0.5, 0.84))
                axes = np.array(fig.axes).reshape((ndim, ndim))
                for i in range(ndim):
                    ax = axes[i, i]
                    ax.axvline(X[i], color="g")
                    ax.axvline(value2[i],color='r')
                # Loop over the histograms
                for yi in range(ndim):
                    for xi in range(yi):
                        ax = axes[yi, xi]
                        ax.axvline(X[xi], color="g")
                        ax.axvline(value2[xi], color="r")
                        ax.axhline(X[yi], color="g")
                        ax.axhline(value2[yi], color="r")
                        ax.plot(X[xi], X[yi], "sg")
                        ax.plot(value2[xi],value2[yi],'sr')
                plt.rcParams.update({'font.size': 12})
                figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_M05_"+str(nsteps*2)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
                fig.savefig(figname)
                fig.clf()
                print('MCMC results maximum Likelihood Point M05:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))

                #--- Plot
                X = np.percentile(samples, 50, axis=0)
                x2_optimized = minimize_age_AV_vector_weighted(X)
                x2_spec, x2_phot = minimize_age_AV_vector_weighted_return_chi2_sep(X)
                chi_square_list.loc[row,'M05_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
                chi_square_list.loc[row,'M05_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
                chi_square_list.loc[row,'x2_M05_MCMC50'] = x2_optimized
                chi_square_list.loc[row,'x2_spectra_M05_MCMC50'] = x2_spec
                chi_square_list.loc[row,'x2_photo_M05_MCMC50'] = x2_phot
                chi_square_list.loc[row,'M05_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
                chi_square_list.loc[row,'M05_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])

                n = len(x)
                fig1 = plt.figure(figsize=(20,10))
                frame1 = fig1.add_axes((.1,.35,.8,.6))
                plt.step(x, y, color='r',lw=3)
                plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
                plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
                model_wave, model_flux =minimize_age_AV_vector_weighted_return_flux(X)[1:]
                plt.plot(model_wave, model_flux, color='k',label='TP-AGB heavy',lw=0.5)
                plt.xlim([2.5e3,1.9e4])
                plt.ylim([0.05, 1.2])
                plt.semilogx()
                plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=22)
                plt.legend(loc='upper right',fontsize=24)
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                
                frame2 = fig1.add_axes((.1,.2,.8,.15))  
                relative_spectra = np.zeros([1,n])
                relative_spectra_err = np.zeros([1,n])
                relative_sigma = np.zeros([1,n])
                index0 = 0
                for wave in x:
                    if y[index0]>0.25 and y[index0]<1.35:
                        index = find_nearest(model_wave, wave);#print index
                        relative_spectra[0, index0] = y[index0]/model_flux[index]
                        relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                        relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                        index0 = index0+1
                plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
                index0 = 0
                for i in range(len(wave_list)):
                    try:
                        index = find_nearest(model_wave, wave_list[i])
                    except:
                        pass
                    plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
                    index0 = index0+1
                plt.xlim([2.5e3,1.9e4])
                plt.semilogx()
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.tick_params(axis='both', which='minor', labelsize=20)
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
                plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
                plt.ylim([-5,5])
                plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
                plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
                figname=current_dir+outcome_dir+plot_dir+'GOODSS_M05_SSP_MCMC50_'+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
                plt.savefig(figname)
                plt.clf()
    except:
        pass

# Test with the new M13 models
    print('____________________M13_________________________ Optimization__________________________')
    bnds = ((0.0, 13.0), (0.0, 4.0))
    X = np.array([galaxy_age, intrinsic_Av])
    try:
        sol_M13 = optimize.minimize(minimize_age_AV_vector_weighted_M13, X, bounds = bnds, method='SLSQP')#, options = {'disp': True})
        print('Optimized M13 weighted reduced chisqure result:', sol_M13)
    except:
        sol_M13 = optimize.minimize(minimize_age_AV_vector_weighted_M13, X, bounds = bnds)#, method='SLSQP')#, options = {'disp': True})
        print('Optimized M13 weighted reduced chisqure result:', sol_M13)    
    X = sol_M13.x
    [age_prior_optimized_M13, AV_prior_optimized_M13] = sol_M13.x
    x2_optimized = minimize_age_AV_vector_weighted_M13(X)
    x2_spec, x2_phot = minimize_age_AV_vector_weighted_M13_return_chi2_sep(X)
    chi_square_list.loc[row,'M13_age_opt'] = X[0]#"{0:.2f}".format(X[0])
    chi_square_list.loc[row,'M13_AV_opt'] = X[1]#"{0:.2f}".format(X[1])
    chi_square_list.loc[row,'x2_M13_opt'] = x2_optimized
    chi_square_list.loc[row,'x2_spectra_M13_opt'] = x2_spec
    chi_square_list.loc[row,'x2_photo_M13_opt'] = x2_phot
    
    #--- Plot
    X = sol_M13.x
    n = len(x)
    fig1 = plt.figure(figsize=(20,10))
    frame1 = fig1.add_axes((.1,.35,.8,.6))
    plt.step(x, y, color='r',lw=3)
    plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
    plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
    model_wave, model_flux =minimize_age_AV_vector_weighted_M13_return_flux(X)[1:]
    plt.plot(model_wave, model_flux, color='g',label='TP-AGB mild',lw=0.5)
    plt.xlim([2.5e3,1.9e4])
    plt.ylim([0.05, 1.2])
    plt.semilogx()
    plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(loc='upper right',fontsize=24)
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
    
    frame2 = fig1.add_axes((.1,.2,.8,.15))  
    relative_spectra = np.zeros([1,n])
    relative_spectra_err = np.zeros([1,n])
    relative_sigma = np.zeros([1,n])
    index0 = 0
    for wave in x:
        if y[index0]>0.25 and y[index0]<1.35:
            index = find_nearest(model_wave, wave);#print index
            relative_spectra[0, index0] = y[index0]/model_flux[index]
            relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
            relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
            index0 = index0+1
    plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
    index0 = 0
    for i in range(len(wave_list)):
        try:
            index = find_nearest(model_wave, wave_list[i])
        except:
            pass
        plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
        index0 = index0+1
    plt.xlim([2.5e3,1.9e4])
    plt.semilogx()
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
    plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
    plt.ylim([-5,5])
    plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
    figname=current_dir+outcome_dir+plot_dir+'GOODSS_M13_SSP_opt_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
    plt.savefig(figname)
    plt.clf()

    with Pool() as pool:
        ndim, nwalkers = 2, 10
        tik = time.clock()
        p0 = [sol_M13.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted_M13, pool=pool)
        sampler.run_mcmc(p0,nsteps, progress=True)
        # print(np.size(samples))

        samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
        samples = samples[(samples[:,0] > age_prior_optimized_M13*0.1) & (samples[:,0] < age_prior_optimized_M13*2.0) & (samples[:,1] < AV_prior_optimized_M13*3.0)]
        # print(np.size(samples))

        tok = time.clock()
        multi_time = tok-tik
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        print('Time to run M13 MCMC:'+str(tok-tik))

    try:
        if samples.size > 1e3 :
            value2=np.percentile(samples, 50, axis=0)
            X = np.percentile(samples, 50, axis=0)
            [std_age_prior_optimized_M13, std_AV_prior_optimized_M13] = np.std(samples, axis=0)
            plt.figure(figsize=(32,32),dpi=100)
            fig = corner.corner(samples,
                 labels=["age(Gyr)", r"$\rm A_V$"],
                 levels=(1-np.exp(-0.5),),
                 truths=[age_prior_optimized_M13, AV_prior_optimized_M13],
                 show_titles=True,title_kwargs={'fontsize':12},
                                quantiles=(0.16,0.5, 0.84))
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(X[i], color="g")
            # Loop over the histograms
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(X[i], color="g")
                ax.axvline(value2[i],color='r')
            # Loop over the histograms
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(X[xi], color="g")
                    ax.axvline(value2[xi], color="r")
                    ax.axhline(X[yi], color="g")
                    ax.axhline(value2[yi], color="r")
                    ax.plot(X[xi], X[yi], "sg")
                    ax.plot(value2[xi],value2[yi],'sr')
            plt.rcParams.update({'font.size': 12})
            figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_M13_"+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
            fig.savefig(figname)
            fig.clf()
            print('Maximum Likelihood Point M13:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))

            #--- Plot
            X = np.percentile(samples, 50, axis=0)
            x2_optimized = minimize_age_AV_vector_weighted_M13(X)
            x2_spec, x2_phot = minimize_age_AV_vector_weighted_M13_return_chi2_sep(X)
            chi_square_list.loc[row,'M13_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
            chi_square_list.loc[row,'M13_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
            chi_square_list.loc[row,'x2_M13_MCMC50'] = x2_optimized
            chi_square_list.loc[row,'x2_spectra_M13_MCMC50'] = x2_spec
            chi_square_list.loc[row,'x2_photo_M13_MCMC50'] = x2_phot
            chi_square_list.loc[row,'M13_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
            chi_square_list.loc[row,'M13_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])            
            n = len(x)

            fig1 = plt.figure(figsize=(20,10))
            frame1 = fig1.add_axes((.1,.35,.8,.6))
            plt.step(x, y, color='r',lw=3)
            plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
            plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
            model_wave,model_flux =minimize_age_AV_vector_weighted_M13_return_flux(X)[1:]
            plt.plot(model_wave, model_flux, color='g',label='TP-AGB mild',lw=0.5)
            plt.xlim([2.5e3,1.9e4])
            plt.ylim([0.05, 1.2])#plt.ylim([ymin,ymax])
            plt.semilogx()
            plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.legend(loc='upper right',fontsize=24)
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
            
            frame2 = fig1.add_axes((.1,.2,.8,.15))  
            relative_spectra = np.zeros([1,n])
            relative_spectra_err = np.zeros([1,n])
            relative_sigma = np.zeros([1,n])
            index0 = 0
            for wave in x:
                if y[index0]>0.25 and y[index0]<1.35:
                    index = find_nearest(model_wave, wave);
                    relative_spectra[0, index0] = y[index0]/model_flux[index]
                    relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                    relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                    index0 = index0+1
            plt.step(x[:index0], relative_sigma[0,:index0], color='r', linewidth=2)
            index0 = 0
            for i in range(len(wave_list)):
                try:
                    index = find_nearest(model_wave, wave_list[i])
                except:
                    pass
                plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
                index0 = index0+1
            plt.xlim([2.5e3,1.9e4])
            plt.semilogx()
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
            plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
            plt.ylim([-5,5])
            plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
            plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
            figname=current_dir+outcome_dir+plot_dir+'GOODSS_M13_SSP_MCMC50_'+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
            plt.savefig(figname)
            plt.clf()
        else:
            with Pool() as pool:
                print('modified steps:',nsteps*2)
                ndim, nwalkers = 2, 10
                tik = time.clock()
                p0 = [sol_M13.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted_M13, pool=pool)
                sampler.run_mcmc(p0,nsteps*2, progress=True)
                samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
                samples = samples[(samples[:,0] > age_prior_optimized_M13*0.1) & (samples[:,0] < age_prior_optimized_M13*2.0) & (samples[:,1] < AV_prior_optimized_M13*3.0)]
                tok = time.clock()
                multi_time = tok-tik
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                print('Time to run M13 MCMC:'+str(tok-tik))
            if samples.size > 1e3 :
                value2=np.percentile(samples, 50, axis=0)
                X = np.percentile(samples, 50, axis=0)
                [std_age_prior_optimized_M13, std_AV_prior_optimized_M13] = np.std(samples, axis=0)
                plt.figure(figsize=(32,32),dpi=100)
                fig = corner.corner(samples,
                     labels=["age(Gyr)", r"$\rm A_V$"],
                     levels=(1-np.exp(-0.5),),
                     truths=[age_prior_optimized_M13, AV_prior_optimized_M13],
                     show_titles=True,title_kwargs={'fontsize':12},
                                    quantiles=(0.16,0.5, 0.84))
                axes = np.array(fig.axes).reshape((ndim, ndim))
                for i in range(ndim):
                    ax = axes[i, i]
                    ax.axvline(X[i], color="g")
                # Loop over the histograms
                for i in range(ndim):
                    ax = axes[i, i]
                    ax.axvline(X[i], color="g")
                    ax.axvline(value2[i],color='r')
                # Loop over the histograms
                for yi in range(ndim):
                    for xi in range(yi):
                        ax = axes[yi, xi]
                        ax.axvline(X[xi], color="g")
                        ax.axvline(value2[xi], color="r")
                        ax.axhline(X[yi], color="g")
                        ax.axhline(value2[yi], color="r")
                        ax.plot(X[xi], X[yi], "sg")
                        ax.plot(value2[xi],value2[yi],'sr')
                plt.rcParams.update({'font.size': 12})
                figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_M13_"+str(nsteps*2)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
                fig.savefig(figname)
                fig.clf()
                print('Maximum Likelihood Point M13:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))

                #--- Plot
                X = np.percentile(samples, 50, axis=0)
                x2_optimized = minimize_age_AV_vector_weighted_M13(X)
                x2_spec, x2_phot = minimize_age_AV_vector_weighted_M13_return_chi2_sep(X)
                chi_square_list.loc[row,'M13_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
                chi_square_list.loc[row,'M13_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
                chi_square_list.loc[row,'x2_M13_MCMC50'] = x2_optimized
                chi_square_list.loc[row,'x2_spectra_M13_MCMC50'] = x2_spec
                chi_square_list.loc[row,'x2_photo_M13_MCMC50'] = x2_phot
                chi_square_list.loc[row,'M13_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
                chi_square_list.loc[row,'M13_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])            
                n = len(x)

                fig1 = plt.figure(figsize=(20,10))
                frame1 = fig1.add_axes((.1,.35,.8,.6))
                plt.step(x, y, color='r',lw=3)
                plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
                plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
                model_wave =minimize_age_AV_vector_weighted_M13_return_flux(X)[1]
                model_flux =minimize_age_AV_vector_weighted_M13_return_flux(X)[2]
                plt.plot(model_wave, model_flux, color='g',label='TP-AGB mild',lw=0.5)
                plt.xlim([2.5e3,1.9e4])
                plt.ylim([0.05, 1.2])
                plt.semilogx()
                plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=22)
                plt.legend(loc='upper right',fontsize=24)
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                
                frame2 = fig1.add_axes((.1,.2,.8,.15))  
                relative_spectra = np.zeros([1,n])
                relative_spectra_err = np.zeros([1,n])
                relative_sigma = np.zeros([1,n])
                index0 = 0
                for wave in x:
                    if y[index0]>0.25 and y[index0]<1.35:
                        index = find_nearest(model_wave, wave);#print index
                        relative_spectra[0, index0] = y[index0]/model_flux[index]
                        relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                        relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                        index0 = index0+1
                plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
                index0 = 0
                for i in range(len(wave_list)):
                    try:
                        index = find_nearest(model_wave, wave_list[i])
                    except:
                        pass
                    plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
                    index0 = index0+1
                plt.xlim([2.5e3,1.9e4])
                plt.semilogx()
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.tick_params(axis='both', which='minor', labelsize=20)
                plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
                plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
                plt.ylim([-5,5])
                plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)

                plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
                figname=current_dir+outcome_dir+plot_dir+'GOODSS_M13_SSP_MCMC50_'+str(nsteps*2)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
                plt.savefig(figname)
                plt.clf()
    except:
        pass

# Test with the new BC03 models
    print('____________________BC03_________________________ Optimization__________________________')
    bnds = ((0.0, 13.0), (0.0, 4.0))
    X = np.array([galaxy_age,intrinsic_Av])
    sol_BC03 = optimize.minimize(minimize_age_AV_vector_weighted_BC03, X, bounds = bnds, method='SLSQP')#, options = {'disp': True})
    print('Optimized BC03 weighted reduced chisqure result:', sol_BC03)
    [age_prior_optimized_BC03, AV_prior_optimized_BC03] = sol_BC03.x
    X = sol_BC03.x
    x2_optimized = minimize_age_AV_vector_weighted_BC03(X)
    x2_spec, x2_phot = minimize_age_AV_vector_weighted_BC03_return_chi2_sep(X)
    chi_square_list.loc[row,'BC_age_opt'] = X[0]#"{0:.2f}".format(X[0])
    chi_square_list.loc[row,'BC_AV_opt'] = X[1]#"{0:.2f}".format(X[1])
    chi_square_list.loc[row,'x2_BC_opt'] = x2_optimized
    chi_square_list.loc[row,'x2_spectra_BC_opt'] = x2_spec
    chi_square_list.loc[row,'x2_photo_BC_opt'] = x2_phot

    #--- Plot
    X = sol_BC03.x
    n = len(x)
    fig1 = plt.figure(figsize=(20,10))
    frame1 = fig1.add_axes((.1,.35,.8,.6))
    plt.step(x, y, color='r',lw=3)
    plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
    plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
    BC03_flux_attenuated = minimize_age_AV_vector_weighted_BC03_mod_no_weight_return_flux(X)[1]
    plt.plot(BC03_wave_list_num, BC03_flux_attenuated, color='orange',label='TP-AGB light',lw=0.5)
    model_wave = BC03_wave_list_num
    model_flux = BC03_flux_attenuated
    plt.xlim([2.5e3,1.9e4])
    plt.ylim([0.05, 1.2])
    plt.semilogx()
    plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(loc='upper right',fontsize=24)
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
    
    frame2 = fig1.add_axes((.1,.2,.8,.15))  
    relative_spectra = np.zeros([1,n])
    relative_spectra_err = np.zeros([1,n])
    relative_sigma = np.zeros([1,n])
    index0 = 0
    for wave in x:
        if y[index0]>0.25 and y[index0]<1.35:
            index = find_nearest(model_wave, wave);
            relative_spectra[0, index0] = y[index0]/model_flux[index]
            relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
            relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
            index0 = index0+1
    plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
    index0 = 0
    for i in range(len(wave_list)):
        try:
            index = find_nearest(model_wave, wave_list[i])
        except:
            pass
        plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
        index0 = index0+1
    plt.xlim([2.5e3,1.9e4])
    plt.semilogx()
    plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
    plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
    plt.ylim([-5,5])
    plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)

    plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
    plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
    plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    figname=current_dir+outcome_dir+plot_dir+'GOODSS_BC03_SSP_opt_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
    plt.savefig(figname)
    plt.clf()

    with Pool() as pool:
        ndim, nwalkers = 2, 10
        tik = time.clock()
        p0 = [sol_BC03.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted_BC03, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)
        # print(np.size(samples))

        samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
        samples = samples[(samples[:,0] > age_prior_optimized_BC03*0.1) & (samples[:,0] < age_prior_optimized_BC03*2.0) & (samples[:,1] < AV_prior_optimized_BC03*3.0)]
        # print(np.size(samples))

        tok = time.clock()
        multi_time = tok-tik
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        print('Time to run BC03 MCMC:'+str(tok-tik))
    try:
        if samples.size > 1e3:
            value2=np.percentile(samples,50,axis=0)
            X = np.percentile(samples, 50,axis=0)
            [std_age_prior_optimized_BC03, std_AV_prior_optimized_BC03] = np.std(samples, axis=0)
            plt.figure(figsize=(32,32),dpi=100)
            fig = corner.corner(samples,
                                labels=["age(Gyr)", r"$\rm A_V$"],\
                                truths=[age_prior_optimized_BC03, AV_prior_optimized_BC03],\
                                levels = (1-np.exp(-0.5),),\
                                show_titles=True,title_kwargs={'fontsize':12},
                                quantiles=(0.16,0.5, 0.84))
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(X[i], color="g")
                ax.axvline(value2[i],color='r')
            # Loop over the histograms
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(X[xi], color="g")
                    ax.axvline(value2[xi], color="r")
                    ax.axhline(X[yi], color="g")
                    ax.axhline(value2[yi], color="r")
                    ax.plot(X[xi], X[yi], "sg")
                    ax.plot(value2[xi],value2[yi],'sr')
            plt.rcParams.update({'font.size': 12})
            figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_BC03_"+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
            fig.savefig(figname)
            fig.clf()
            print('Maximum Likelihood Point BC03:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))
            
            #--- Plot
            X = np.percentile(samples, 50, axis=0)
            x2_optimized = minimize_age_AV_vector_weighted_BC03(X)
            x2_spec, x2_phot = minimize_age_AV_vector_weighted_BC03_return_chi2_sep(X)
            chi_square_list.loc[row,'BC_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
            chi_square_list.loc[row,'BC_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
            chi_square_list.loc[row,'x2_BC_MCMC50'] = x2_optimized
            chi_square_list.loc[row,'x2_spectra_BC_MCMC50'] = x2_spec
            chi_square_list.loc[row,'x2_photo_BC_MCMC50'] = x2_phot
            chi_square_list.loc[row,'BC_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
            chi_square_list.loc[row,'BC_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])
            n = len(x)
            
            fig1 = plt.figure(figsize=(20,10))
            frame1 = fig1.add_axes((.1,.35,.8,.6))
            plt.step(x, y, color='r',lw=3)
            plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
            plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
            BC03_flux_attenuated = minimize_age_AV_vector_weighted_BC03_mod_no_weight_return_flux(X)[1]
            plt.plot(BC03_wave_list_num, BC03_flux_attenuated, color='orange',label='TP-AGB light',lw=0.5)
            model_wave = BC03_wave_list_num
            model_flux = BC03_flux_attenuated            
            plt.xlim([2.5e3,1.9e4])
            plt.ylim([0.05, 1.2])
            plt.semilogx()
            plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.legend(loc='upper right',fontsize=24)
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
            
            frame2 = fig1.add_axes((.1,.2,.8,.15))  
            relative_spectra = np.zeros([1,n])
            relative_spectra_err = np.zeros([1,n])
            relative_sigma = np.zeros([1,n])
            index0 = 0
            for wave in x:
                if y[index0]>0.25 and y[index0]<1.35:
                    index = find_nearest(model_wave, wave);#print index
                    relative_spectra[0, index0] = y[index0]/model_flux[index]
                    relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                    relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                    index0 = index0+1
            plt.step(x[:index0], relative_sigma[0,:index0], color='r', linewidth=2)
            plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)

            index0 = 0
            for i in range(len(wave_list)):
                try:
                    index = find_nearest(model_wave, wave_list[i])
                except:
                    pass
                index0 = index0+1
            plt.xlim([2.5e3,1.9e4])
            plt.semilogx()
            plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
            plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
            plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
            plt.ylim([-5,5])
            plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
            plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
            plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
            plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            figname=current_dir+outcome_dir+plot_dir+'GOODSS_BC03_SSP_MCMC50_'+str(nsteps)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
            plt.savefig(figname)
            plt.clf()
        else:
            with Pool() as pool:
                print('modified steps:',nsteps)
                ndim, nwalkers = 2, 10
                tik = time.clock()
                p0 = [sol_BC03.x + 4.*np.random.rand(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lg_minimize_age_AV_vector_weighted_BC03, pool=pool)
                sampler.run_mcmc(p0, nsteps*2, progress=True)
                samples = sampler.chain[:, 500:, :].reshape((-1,ndim))
                samples = samples[(samples[:,0] > age_prior_optimized_BC03*0.1) & (samples[:,0] < age_prior_optimized_BC03*2.0) & (samples[:,1] < AV_prior_optimized_BC03*3.0)]
                tok = time.clock()
                multi_time = tok-tik
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                print('Time to run BC03 MCMC:'+str(tok-tik))
            if samples.size > 1e3:
                value2=np.percentile(samples,50,axis=0)
                X = np.percentile(samples,50,axis=0)
                [std_age_prior_optimized_BC03, std_AV_prior_optimized_BC03] = np.std(samples, axis=0)
                plt.figure(figsize=(32,32),dpi=100)
                fig = corner.corner(samples,
                                    labels=["age(Gyr)", r"$\rm A_V$"],\
                                    truths=[age_prior_optimized_BC03, AV_prior_optimized_BC03],\
                                    levels = (1-np.exp(-0.5),),\
                                    show_titles=True,title_kwargs={'fontsize':12},
                                    quantiles=(0.16,0.5, 0.84))
                axes = np.array(fig.axes).reshape((ndim, ndim))
                for i in range(ndim):
                    ax = axes[i, i]
                    ax.axvline(X[i], color="g")
                    ax.axvline(value2[i],color='r')
                # Loop over the histograms
                for yi in range(ndim):
                    for xi in range(yi):
                        ax = axes[yi, xi]
                        ax.axvline(X[xi], color="g")
                        ax.axvline(value2[xi], color="r")
                        ax.axhline(X[yi], color="g")
                        ax.axhline(value2[yi], color="r")
                        ax.plot(X[xi], X[yi], "sg")
                        ax.plot(value2[xi],value2[yi],'sr')
                plt.rcParams.update({'font.size': 12})
                figname=current_dir+outcome_dir+plot_dir+"goodss_triangle_BC03_"+str(nsteps*2)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+".pdf"
                fig.savefig(figname)
                fig.clf()
                print('Maximum Likelihood Point BC03:', np.percentile(samples, 50, axis=0), np.std(samples, axis=0))
                
                #--- Plot
                X = np.percentile(samples, 50, axis=0)
                x2_optimized = minimize_age_AV_vector_weighted_BC03(X)
                x2_spec, x2_phot = minimize_age_AV_vector_weighted_BC03_return_chi2_sep(X)
                chi_square_list.loc[row,'BC_age_MCMC50'] = X[0]#"{0:.2f}".format(X[0])
                chi_square_list.loc[row,'BC_AV_MCMC50'] = X[1]#"{0:.2f}".format(X[1])
                chi_square_list.loc[row,'x2_BC_MCMC50'] = x2_optimized
                chi_square_list.loc[row,'x2_spectra_BC_MCMC50'] = x2_spec
                chi_square_list.loc[row,'x2_photo_BC_MCMC50'] = x2_phot
                chi_square_list.loc[row,'BC_age_std'] = np.std(samples, axis=0)[0]#"{0:.2f}".format(np.std(samples, axis=0)[0])
                chi_square_list.loc[row,'BC_AV_std'] = np.std(samples, axis=0)[1]#"{0:.2f}".format(np.std(samples, axis=0)[1])
                n = len(x)
                
                fig1 = plt.figure(figsize=(20,10))
                frame1 = fig1.add_axes((.1,.35,.8,.6))
                plt.step(x, y, color='r',lw=3)
                plt.fill_between(x,(y+y_err),(y-y_err),alpha=0.1)
                plt.errorbar(wave_list, photometric_flux, xerr=band_list, yerr=photometric_flux_err_mod, color='r', fmt='o', label='photometric data', markersize='14')
                BC03_flux_attenuated = minimize_age_AV_vector_weighted_BC03_mod_no_weight_return_flux(X)[1]
                plt.plot(BC03_wave_list_num, BC03_flux_attenuated, color='orange',label='TP-AGB light',lw=0.5)
                model_wave = BC03_wave_list_num
                model_flux = BC03_flux_attenuated            
                plt.xlim([2.5e3,1.9e4])
                plt.ylim([0.05, 1.2])
                plt.semilogx()
                plt.ylabel(r'$\rm F_{\lambda}/F_{0.55\mu m}$',fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=22)
                plt.legend(loc='upper right',fontsize=24)
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                
                frame2 = fig1.add_axes((.1,.2,.8,.15))  
                relative_spectra = np.zeros([1,n])
                relative_spectra_err = np.zeros([1,n])
                relative_sigma = np.zeros([1,n])
                index0 = 0
                for wave in x:
                    if y[index0]>0.25 and y[index0]<1.35:
                        index = find_nearest(model_wave, wave);#print index
                        relative_spectra[0, index0] = y[index0]/model_flux[index]
                        relative_spectra_err[0, index0] = y_err[index0]/model_flux[index]
                        relative_sigma[0, index0] = (y[index0]-model_flux[index])/y_err[index0]
                        index0 = index0+1
                plt.step(x[:index0], relative_sigma[0, :index0], color='r', linewidth=2)
                index0 = 0
                for i in range(len(wave_list)):
                    try:
                        index = find_nearest(model_wave, wave_list[i])
                    except:
                        pass
                    plt.errorbar(wave_list[i], (photometric_flux[i]-model_flux[index])/photometric_flux_err_mod[i], xerr=band_list[i], fmt='o', color='r', markersize=12)
                    index0 = index0+1
                plt.xlim([2.5e3,1.9e4])
                plt.semilogx()
                plt.axhline(3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(-3.0, linestyle='--', linewidth=1, color='k')
                plt.axhline(1.0, linestyle='--', linewidth=0.5, color='k')
                plt.axhline(-1.0, linestyle='--', linewidth=0.5, color='k')
                plt.ylim([-5,5])
                plt.ylabel(r'$\rm (F_{\lambda,\rm data}-F_{\lambda,\rm model})/F_{\lambda,\rm err}$',fontsize=16)
                plt.xlabel(r'Wavelength($\rm \AA$)', fontsize=20)
                plt.axvspan(1.06e4,1.08e4, color='gray',alpha=0.1)
                plt.axvspan(1.12e4,1.14e4, color='gray',alpha=0.1)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.tick_params(axis='both', which='minor', labelsize=20)
                figname=current_dir+outcome_dir+plot_dir+'GOODSS_BC03_SSP_MCMC50_'+str(nsteps*2)+'_'+str(region)+'_'+str(ID)+'_'+"{0:.2f}".format(X[0])+'Gyr_AV'+"{0:.2f}".format(X[1])+'.pdf'
                plt.savefig(figname)
                plt.clf()
    except:
        pass

    chi2_array=chi_square_list.loc[i,['x2_M05_opt','x2_M13_opt','x2_BC_opt','x2_M05_MCMC50','x2_M13_MCMC50','x2_BC_MCMC50']].values
    AV_array=chi_square_list.loc[i,['M05_AV_opt','M13_AV_opt','BC_AV_opt','M05_AV_MCMC50','M13_AV_MCMC50','BC_AV_MCMC50']].values
    index = find_nearest(chi2_array,0)
    AV_opt = float(AV_array[index])
    spectra_extinction = calzetti00(x, AV_opt, 4.05)
    spectra_flux_correction = 10 ** (-0.4 * spectra_extinction)
    y_corr = y / spectra_flux_correction
    if redshift<0.49:
        try:
            chi_square_list.loc[row,'grism_index_AV_corr'] = Lick_index_ratio(x,y_corr)
        except:
            pass
    chi_square_list.to_csv('/home/siqi/TAPS/TAPS/outcome/numeric/chi_square_list_goodss_'+str(date)+'_photo_'+str(region)+'_'+str(ID)+'.csv')

