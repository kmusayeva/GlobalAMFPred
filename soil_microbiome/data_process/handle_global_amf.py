"""
From species abundancy files from Global AMF data base, it creates a single file
containing environmental variables and all species abundancies.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from soil_microbiome.utils.utils import *
from .. import global_vars

directory = os.path.join(global_vars['global_amf_dir'])
cols_numeric = ['longitude', 'latitude', 'MAT', 'MAP', 'pH']
dfs = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        df = pd.read_csv(os.path.join(directory, filename), delimiter='\t')  # Adjust delimiter if necessary
        # Get the file name without the extension
        species_name = os.path.splitext(filename)[0]
        # Rename the 'abundances' column to the species name
        rows = df[cols_numeric].apply(lambda x: x.astype(str).str.contains('NA_', na=False)).any(axis=1)
        df = df[~rows]
        df.rename(columns={'abundances': species_name}, inplace=True)
        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.replace("NA_", np.nan, inplace=True)
combined_df[cols_numeric] = combined_df[cols_numeric].apply(pd.to_numeric)
combined_df_grouped = combined_df.groupby(combined_df.columns[0:16].tolist(), as_index=False)[
    combined_df.columns[16:]].agg('sum')
combined_df_grouped.reset_index(inplace=True)

# Biome stands for grassland, forest, cropland, etc...
df = pd.concat([pd.get_dummies(combined_df_grouped['Biome']), combined_df_grouped], axis=1)
biomes = combined_df_grouped['Biome'].unique().tolist()

df2 = df.groupby(['longitude', 'latitude'], as_index=False).agg({
    'pH': 'mean',
    'MAT': 'mean',
    'MAP': 'mean',
    **{biome: 'mean' for biome in biomes},
    'paper_id': 'first',
    'continent': 'first',
    **{col: 'sum' for col in df.columns[26:]}
})

df2[biomes] = df2[biomes].applymap(lambda x: 1 if x > 0 else x)
file = os.path.join(directory, "Species.xlsx")
df.to_excel(file, index=False)
