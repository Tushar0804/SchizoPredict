import pandas as pd
import glob

path = "/Users/tusharsharma/B Tech/RESEARCH PROJECTS/Schizo/HO5"

HO_file_list = glob.glob(path + "/HO*.xlsx")
SO_file_list = glob.glob(path + "/SO*.xlsx")

# Combining all HO files first
HO =[]
for file in HO_file_list:
    HO.append(pd.read_excel(file))

HO_Final = pd.DataFrame()

for file in HO: 
    df = pd.DataFrame(file)    
    filtered_df = df[df['Time'].astype(str).str.contains(r'^\d+\.\d{1}$')]    
    HO_Final = HO_Final.append(df, ignore_index=True)

HO_Final['Class'] = "HO"
HO_Final.to_csv("HO_Final.csv", index=False)

# Combining all SO files
SO =[]
for file in SO_file_list:
    SO.append(pd.read_excel(file))

SO_Final = pd.DataFrame()

for file in SO: 
    df = pd.DataFrame(file)
    # filtered_df = df[df['Time'].astype(str).str.contains(r'^\d+\.\d{1}$')]   
    SO_Final = SO_Final.append(df, ignore_index=True)

SO_Final = SO_Final.drop('class', axis=1)
SO_Final['Class'] = "SO"
SO_Final.to_csv("SO_Final.csv", index=False)

# Final Merging
df = pd.DataFrame()
df = df.append(HO_Final)
df = df.append(SO_Final)

df.to_csv("dataset.csv", index=False)
print("DONE")