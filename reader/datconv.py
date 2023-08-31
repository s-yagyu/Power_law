"""
AC Series Data Converter

"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

def getEncode(filepath):
    """Character code automatic judgment function
     If the file name contains Japanese, the encoding method may be Shift-jis.
     Function that returns the encoding method.

    Args:
        filepath (str): fil path and name

    Returns:
        encode type(str): encode type
    """
    encs = "iso-2022-jp euc-jp shift_jis utf-8".split()
    for enc in encs:
        with open(filepath, encoding=enc) as fr:
            try:
                fr = fr.read()
            except UnicodeDecodeError:
                continue
        return enc


class AcConv():
    """
    Extraction of measurement metadata and measurement data from the AC.dat file.
    
    args:
        file_name (str): .dat filename

    Example : 
        file_name = r'./datafile.dat'
        acdata = AcConv(file_name)
        acdata.convert
        
        print('metadata')
        print(acdata.metadata)
        
        print('metadata json-type')
        print(acdata.json)
   
        print('pys data by dataframe type')
        ##### "uvEnergy", "countingCorrection", "photonCorrection", "yield", "nyield", 
        #####  "guideline", "countingRate","flGrandLevel","flRegLevel","uvIntensity"
        print(acdata.df)
        
        print("user analysis values")
        print(acdata.estimate_value)
        
        print('export dataframe data to csv')
        acdata.export_df2csv()
    
    
    """

    def __init__(self,file_name):
        self.file_name = Path(file_name)
        
    def convert(self,json_export=False):
        """
        Attribution
        
        - original metadata keys
        
        "fileType","deadTime","countingTime","powerNumber","anodeVoltage",
        "step","model","yAxisMaximum","startEnergy",
        "finishEnergy","flagDifDataGroundLevel","bgCountingRate",
        "measureDate","sampleName",
        "uvIntensity59","targetUv","nameLightCorrection","sensitivity1","sensitivity2"

        - original data coloum keys
        
        "uvEnergy", "countingRate", "flGrandLevel", "flRegLevel","uvIntensity"
 
        - file optional keys
        
        "countingCorrection", "photonCorrection", "pyield", "npyield","nayield", "guideline"
        
        "estimate_value" include ('thresholdEnergy', 'slope', 'yslice', 'bg')
        
        
        """
        _ = self._read_para()
        self.countingCorrection = self._count_calibration()
        self.photonCorrection = self._photon_calibration()
        self.ydata, self.npyield = self._pyield_intensity()
        _ = self.user_estimation()
        
        meta_keys = ["fileType","deadTime","countingTime","powerNumber",
                      "anodeVoltage","step","model","yAxisMaximum","startEnergy",
                      "finishEnergy","flagDifDataGroundLevel","bgCountingRate","measureDate","sampleName",
                      "uvIntensity59","targetUv","nameLightCorrection","sensitivity1","sensitivity2"]
        
        meta_values = [self.fileType, self.deadTime, self.countingTime, self.powerNumber,
                      self.anodeVoltage, self.step,self.model, self.yAxisMaximum, self.startEnergy,
                      self.finishEnergy,self.flagDifDataGroundLevel, self.bgCountingRate,self.measureDate,self.sampleName,
                      self.uvIntensity59,self.targetUv, self.nameLightCorrection, self.sensitivity1,self.sensitivity2]
        
        calc_data_keys = ["uvEnergy", "countingCorrection", "photonCorrection", "pyield", "npyield","nayield", "guideline",
                         "countingRate","flGrandLevel","flRegLevel","uvIntensity"]
        
        calc_data_values = [self.uvEnergy, self.countingCorrection, self.photonCorrection,self.ydata, self.npyield, self.nayield, self.guideline,
                            self.countingRate,self.flGrandLevel,self.flRegLevel,self.uvIntensity]
        calc_data_values_list = [d.tolist() for d in calc_data_values]
        
        file_meta ={'file_name':self.file_name.name}
        
        self.metadata = dict(zip(meta_keys + calc_data_keys, meta_values + calc_data_values_list)) 
        self.metadata.update(self.estimate_value)
        self.metadata.update(file_meta)
        
        self.calcdata = dict(zip(calc_data_keys,calc_data_values))
        
        # self.metadata_wo_calc means that the calculated data, which is array data, is not included.
        self.metadata_wo_calc = dict(zip(meta_keys,meta_values)) 
        self.metadata_wo_calc.update(self.estimate_value)
        self.metadata_wo_calc.update(file_meta)
        
        
        self.json = self._json_out(self.metadata)
        self.df = self._df_out(self.calcdata)
        
    def _read_para(self):
        # read parameters up to the third line
        enc = getEncode(str(self.file_name))
        with open(str(self.file_name),encoding=enc) as f:
            reader = csv.reader(f)
            meta = [row for row in reader]

        # Match the parameter list to the full parameter list.
        # this case is AC-5 old dat type
        if len(meta[0]) == 10:    
            meta[0].extend(['0','0.0'])
            meta[2].extend(['1','1'])
            
        else:
            pass
        
        self.fileType = meta[0][0]
        self.deadTime = float(meta[0][1])
        self.countingTime = float(meta[0][2])
        self.powerNumber = float(meta[0][3])
        self.anodeVoltage = float(meta[0][4])
        self.step = float(meta[0][5])
        self.model = meta[0][6]
        self.yAxisMaximum = float(meta[0][7])
        self.startEnergy = float(meta[0][8])
        self.finishEnergy = float(meta[0][9])
        self.flagDifDataGroundLevel = int(meta[0][10])
        self.bgCountingRate = float(meta[0][11])
        self.measureDate = meta[1][0]
        self.sampleName = meta[1][1]
        self.uvIntensity59 = float(meta[2][0])
        self.targetUv = float(meta[2][1])
        self.nameLightCorrection = meta[2][2]
        self.sensitivity1 = float(meta[2][3])
        self.sensitivity2 = float(meta[2][4])
        
        # Transpose row to col
        raw_data = [list(x) for x in zip(*meta[3:])]
        self.uvEnergy = np.array([float(v) for v in raw_data[0]])
        self.countingRate = np.array([float(v) for v in raw_data[1]])
        self.flGrandLevel = np.array([int(v) for v in raw_data[2]])
        self.flRegLevel = np.array([int(v) for v in raw_data[3]])
        self.uvIntensity = np.array([float(v) for v in raw_data[4]])
       
       
    def _count_calibration(self):
        """
        count_calibration
        "AC-3" and "AC-2" data do not require count calibration.
        """
        # The "AC-3" and "AC-2" counts do not need to be calibrated.
        if self.model == 'AC-3' or self.model == 'AC-2':
            # print('AC-2,AC-3')
            self.countingCorrection = self.countingRate
            
        else:
            part1 = (self.countingRate)/(1-self.deadTime*(self.countingRate))
            part2 = np.exp(0.13571/(1-0.0028*(self.countingRate)))*self.sensitivity1
            part3 = (self.bgCountingRate)/(1-self.deadTime*(self.bgCountingRate))
            part4 = np.exp(0.13571/(1-0.0028*(self.bgCountingRate)))*self.sensitivity1
            self.countingCorrection = part1*part2-part3*part4
          
        return self.countingCorrection

    def _photon_calibration(self):
        
        # number of Photons
        self.nPhoton = 0.625*(self.uvIntensity/self.uvEnergy)
        # number of photons per unit 
        self.unitPhoton = (self.uvIntensity59*0.625)/5.9
        self.photonCorrection = self.nPhoton/self.unitPhoton

        return self.photonCorrection

    def _pyield_intensity(self):
        self.ydata = self.countingCorrection/self.photonCorrection
        self.npyield = np.power(self.ydata, self.powerNumber)
        
        # replace Nan to 0
        self.ydata = np.where(self.ydata < 0, 0, self.ydata)
        # replace Nan to 0
        self.npyield[np.isnan(self.npyield)] = 0

        return self.ydata, self.npyield
    
    
    @staticmethod
    @np.vectorize
    def relu(xdata, a, b, bg):
        ip = (bg - b)/a
        u = (xdata - ip)
        return a * u * (u > 0.0) + bg

    
    @staticmethod
    def user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False):
        
        bg = np.nanmean(bg_ydata)
        popt = np.polyfit(reg_xdata, reg_ydata, 1)

        a = popt[0]
        b = popt[1]
        
        cross_point =  (bg - b) / a

        if printf:
            print(f"bg:{bg}, a(slope):{a}, b(yslice):{b}")
            print(f"thresholdEnergy -> {cross_point}" )
            
        return {'thresholdEnergy': cross_point, 'slope': a,'yslice': b, 'bg':bg}
        
    def user_estimation(self):
        """ user analized  threshold value

        Returns:
            dict[float]: 'thresholdEnergy', 'slope','yslice','bg'
        """
        # self.df_data =  self.convert_df()

        self.bg_flag_ind = np.where(self.flGrandLevel == -1)[0].tolist()
        self.reg_flag_ind = np.where(self.flRegLevel == -1)[0].tolist()
        # print( bg_flag_ind, reg_flag_ind)
        
        if  self.bg_flag_ind != [] and self.reg_flag_ind != []:
            # back ground difference caribration
            if self.flagDifDataGroundLevel == -1:
                # average
                bg_ave = np.nanmean(self.ydata[self.bg_flag_ind])
                c_pys = self.ydata - bg_ave
                self.cc_pys = np.where(c_pys < 0, 0, c_pys)
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)
                
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                bg_ydata = np.array([0.0]*len(self.bg_flag_ind))
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
                # print(cc_pys)
            
            #  elif self.flagDifDataGroundLevel == 0:  
            else:   
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                
                self.cc_pys = self.ydata
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
                bg_ydata = self.cc_npys[self.bg_flag_ind]
                

            self.estimate_value = AcConv.user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False) 
            self.nayield = self.cc_npys
            self.guideline = AcConv.relu(xdata=self.uvEnergy, a=self.estimate_value['slope'], 
                                        b=self.estimate_value['yslice'], 
                                        bg=self.estimate_value['bg'])
        
        else:
            self.estimate_value =  {'thresholdEnergy': np.nan, 'slope': np.nan,'yslice':np. nan,'bg':np.nan}
            self.nayield = self.npyield
            self.guideline = np.array([np.nan]*len(self.uvEnergy.tolist()))

            
    

    def _json_out(self, dict_metadata,export=False):
        
        if export:
            if jsonfile_name is None:
                jsonfile_name =self.file_name.with_suffix('.json')

            with open(jsonfile_name, 'w') as f:
                json.dump(dict_metadata, f, indent=4)
        
        json_meta = json.dumps(dict_metadata)
        
        return json_meta

        
    def _df_out(self,dict_data):
        
        df_temp = pd.DataFrame(dict_data)
    
        return df_temp
      
    def export_df2csv(self,df_out_file_name=None):
        """export Dataframe data to csv

        Args:
            df_out_file_name (str, optional):output filename. Defaults to None.
        """
        if df_out_file_name is None:
            df_out_file_name =self.file_name.with_suffix('.csv')
        
        self.df.to_csv(df_out_file_name, index=False)
    
    def export_json(self,json_out_file_name=None):
        """export json

        Args:
            json_out_file_name (str, optional):output filename. Defaults to None.
        """
        if json_out_file_name is None:
            json_out_file_name =self.file_name.with_suffix('.json')

        dict_json = json.loads(self.json)
        with open(json_out_file_name, 'w') as f:
            json.dump(dict_json, f, indent=4)


class AdvAcConv(AcConv):
    
    def _xy_convert(self):
        # Overflowのデータは0になっている。
        def find_first_positive_index(arr):
            """後ろから探索して最初の正の数のインデックスを取得
            Args:
                arr (ndarray): ndarray

            Returns:
                int: index
                
            # テスト用の配列
            array = np.array([-3, -2, 0, -1, 4, 5, -6])
            print(len(array))
            index = find_first_positive_index(array)
            if index != -1:
                print("最初の正の数はインデックス", index, "にあります。")
            else:
                print("正の数は見つかりませんでした。")
            
            >>>最初の正の数はインデックス 5 にあります。
            
            """
            for i in range(len(arr)-1, -1, -1):
                if arr[i] > 0:
                    return i
            return -1  # 正の数が見つからなかった場合
        
        index = find_first_positive_index(self.ydata)
        # print(index)
        if index != len(self.ydata)-1:
            self.xdata = self.xdata[:index]
            self.ydata = self.ydata[:index]

        # もう一度繰り返す
        index = find_first_positive_index(self.ydata)
        # print(index)
        if index != len(self.ydata)-1:
            self.xdata = self.xdata[:index]
            self.ydata = self.ydata[:index]
    
    def flag_index(self):
        self.bg_flag_ind = np.where(self.flGrandLevel == -1)[0].tolist()
        self.reg_flag_ind = np.where(self.flRegLevel == -1)[0].tolist()
        # print( bg_flag_ind, reg_flag_ind)
        
        # if  bg_flag_ind != [] and reg_flag_ind != []:
        return self.bg_flag_ind, self.reg_flag_ind
    
    def user_estimation_23(self, powerN):

        """ user analized  threshold value
        
        args:
            powerN (float): 1/n  example: 1/2

        Returns:
            dict[float]: 'thresholdEnergy', 'slope','yslice','bg'
        """
        # self.df_data =  self.convert_df()
        bg_flag_ind = np.where(self.flGrandLevel == -1)[0].tolist()
        reg_flag_ind = np.where(self.flRegLevel == -1)[0].tolist()
        # print( bg_flag_ind, reg_flag_ind)
        
        if  bg_flag_ind != [] and reg_flag_ind != []:
            # back ground difference caribration
            if self.flagDifDataGroundLevel == -1:
                # average
                bg_ave = np.nanmean(self.ydata[bg_flag_ind])
                c_pys = self.ydata - bg_ave
                self.cc_pys = np.where(c_pys < 0, 0, c_pys)
                self.cc_npys = np.power(self.cc_pys, powerN)
                
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                bg_ydata = np.array([0.0]*len(bg_flag_ind))
                reg_xdata = self.uvEnergy[reg_flag_ind]
                reg_ydata = self.cc_npys[reg_flag_ind]
                # print(cc_pys)
            
            #  elif self.flagDifDataGroundLevel == 0:  
            else:   
                # bg_xdata = self.uvEnergy[bg_flag_ind]
                
                self.cc_pys = self.ydata
                self.cc_npys = np.power(self.cc_pys, powerN)
                reg_xdata = self.uvEnergy[reg_flag_ind]
                reg_ydata = self.cc_npys[reg_flag_ind]
                bg_ydata = self.cc_npys[bg_flag_ind]
                

            estimate_value = AcConv.user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False) 
            nayield = self.cc_npys
            guideline = AcConv.relu(xdata=self.uvEnergy, a=estimate_value['slope'], 
                                        b=estimate_value['yslice'], 
                                        bg=estimate_value['bg'])
        
        else:
            estimate_value =  {'thresholdEnergy': np.nan, 'slope': np.nan,'yslice':np. nan,'bg':np.nan}
            nayield = self.npyield
            guideline = np.array([np.nan]*len(self.uvEnergy.tolist()))

      
        return {'est_val':estimate_value, 'rex':self.uvEnergy, 'rey':nayield, 'fit':guideline}   
        
            

if __name__ =='__main__':
    pass
    # fp=r'////.dat'
    # fpdata = AcConv(fp) 
    # fpdata.convert()
    # print(fpdata.df)
    # print(fpdata.json)
    # print(fpdata.metadata)
   