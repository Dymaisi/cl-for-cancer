import pandas as pd

# delimiter is \t
data = pd.read_csv('data/CPCGEA-FPKM.csv',delimiter='\t')

# drop the column whose header contains the letter "N"
df_CPCGEA = data[ [c for c in data.columns if not c.startswith('N')] ]

# save changes to another file
df_CPCGEA.to_csv('data/CPCGEA-FPKM_Tumor.csv',index = False)

df_TCGA = pd.read_excel('data/TCGA-PRAD.htseq_FPKM.xlsx')
df_all = pd.read_excel('data/CPGEA-TCGA 20221207 Dr. ZWH.xlsx') 

TXXX_list = []

# get 'TXXX' from headers 'TXXX_WTS' in df_CPCGEA to match with the 'Sample_ID' value in df_all
for col in df_CPCGEA.columns:
    TXXX_list.append(col.split('_')[0])

# find the corresponding patient data and save to another file
matches_df = df_all[df_all['Sample_ID'].isin(TXXX_list)]
matches_df.to_excel('data/CPGEA.xlsx', index=False)

# the same to df_TCGA
for col in df_TCGA.columns:
    TXXX_list.append(col.strip('A'))

matches_df = df_all[(df_all['Sample_ID']).isin(TXXX_list)]
matches_df.to_excel('data/TCGA.xlsx', index=False)

# delete the extra data in df_TCGA
TCGA = matches_df

matches_df = df_TCGA[ [col for col in df_TCGA.columns.tolist() if (col.strip('A') in TCGA['Sample_ID'].tolist()) ] ]

matches_df.to_excel('data/TCGA-PRAD-FPKM-MATCH.xlsx', index=False)