# %% [markdown]
# 1-indexed

# %%
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append()


dataset =  'icews18'

def process():
    task = "1-indexed"
    save_path = f'data/raw/{dataset}/{task}'
    os.makedirs(save_path, exist_ok=True)
    df = pd.read_csv(f"../../../../data/{dataset}_from_original.csv")

    # %% [markdown]
    # Create train

    # %%
    df = df.dropna()


    # %%
    a_a_df = df[['actor1_i', 'actor2_i']].copy()


    # %%
    # def factorize_df(df):
    #     codes, uniques = pd.factorize(df.values.ravel())
    #     res = pd.DataFrame(codes.reshape(df.shape), columns=df.columns)
    #     return codes, uniques, res

    # codes, uniques, a_a_df = factorize_df(a_a_df)
    # mapping = {unique: code for code, unique in enumerate(uniques)}

    # %%
    codes, uniques = pd.factorize(a_a_df['actor1_i'])
    codes += 1
    a_a_df['actor1_i'] = codes
    a_a_df['actor2_i'] = pd.factorize(a_a_df['actor2_i'])[0] + 1
    # a_a_df['actor2_i'] = pd.factorize(a_a_df['actor2_i'])[0]

    mapping = {unique: code+1 for code, unique in enumerate(uniques)}
    # mapping = {unique: code for code, unique in enumerate(uniques)}

    # %%
    mapping

    # %%
    a_a_df['weight'] = 1

    # %%
    a_a_df

    # %%
    a_ac_df = df[['actor1_i', 'cameo_code_i']].copy()
    a_ac_df['actor1_i'] = a_ac_df['actor1_i'].map(lambda x: mapping[x])
    codes, uniques = pd.factorize(a_ac_df['cameo_code_i'].values.ravel())
    codes += 1
    mapping_c = {unique: code+1 for code, unique in enumerate(uniques)}
    # mapping_c = {unique: code for code, unique in enumerate(uniques)}
    a_ac_df['cameo_code_i'] = a_ac_df['cameo_code_i'].map(lambda x: mapping_c[x])

    # %%
    a_ac_df

    # %%
    a_sec_df = df[['actor1_i', 'Source Sectors']].copy()
    a_sec_df['actor1_i'] = a_sec_df['actor1_i'].map(lambda x: mapping[x])
    a_sec_df['source_sectors_i'] = pd.factorize(a_sec_df['Source Sectors'])[0]+1
    # a_sec_df['source_sectors_i'] = pd.factorize(a_sec_df['Source Sectors'])[0]
    a_sec_df = a_sec_df.drop('Source Sectors', axis=1)

    # %%
    a_sec_df

    # %%
    -1 in a_sec_df.actor1_i

    # %%
    a_a_df

    # %%
    a_a_df.to_csv(f"{save_path}/actor_actor.csv", header=False, index=False)

    # %%
    f"{save_path}/actor_action.csv"

    # %%
    a_ac_df.to_csv(f"{save_path}/actor_action.csv", header=False)

    # %% [markdown]
    # 

    # %%
    a_sec_df.to_csv(f"{save_path}/actor_sector.csv", header=False)

# %%

    total_unique = len(np.unique(a_a_df.values.ravel()))
    print("Total actors",total_unique)

    # %%
    print("Total actor 1",a_a_df.actor1_i.nunique())

    # %%
    print("Total actor 2",a_a_df.actor2_i.nunique())

    # %%
    print("Total actions",a_ac_df.cameo_code_i.nunique())

    # %%
    print("Total sectors",a_sec_df.source_sectors_i.nunique())

    # %%


if __name__ == "__main__":
    process()
