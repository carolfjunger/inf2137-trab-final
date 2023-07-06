import pandas as pd
import numpy as np
import os

from constants import COLUNMS, STRING_COLUNMS

class LoaderText:
    
  def load_data(self):
    """loads true and fake news metadata form text files

    Returns:
        Dataframe: dataframe with text metadata
    """
    # concat index
    df_true = self.__create_dataset("true")
    df_fake = self.__create_dataset("fake")
    df_orig = pd.concat([df_true, df_fake])
    df_orig.reset_index(inplace=True) # para o index ser único
    return df_orig
  
  def __create_dataset(self, label):
    """_summary_

    Args:
        label (string): label of the dataset true/fake

    Returns:
        Dataframe: dataframe labeled with text metadata
    """
    total_lines = 3602
    df = pd.DataFrame(columns=COLUNMS, data=np.empty([total_lines, len(COLUNMS)]))
    for i in range(0,total_lines):
        file_name = f"{i+1}-meta.txt"
        file_path = f"full_texts/{label}-meta-information/{file_name}"
        if (os.path.exists(file_path)):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
                df.loc[i,COLUNMS] = lines
        else:
            df.drop(index=i, inplace=True)
    df["label"] = df["link"].apply(lambda x: 1 if label == "true" else 0)
    return df
  
  def get_number_coluns(self, df):
    """

    Args:
        df (Dataframe): complete Dataframe

    Returns:
        list: number colunms as list
    """
    return list(set(df.columns) - set(STRING_COLUNMS))
  
  def get_df_with_transform_types(self, df):
    """get df with number types typed as float

    Args:
        df (dataframe): complete Dataframe 

    Returns:
        dataframe: df with number colunms typed as float
    """
    number_coluns = self.get_number_coluns(df)
    df.replace("None", np.nan, inplace=True)
    df[number_coluns] = df[number_coluns].astype(float)
    return df