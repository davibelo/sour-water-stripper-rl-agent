import numpy as np
import pandas as pd
from scipy.stats import qmc
from pathlib import Path

N = 1000
d = 7
seed = 123
NUM_FILES = 2

sampler = qmc.LatinHypercube(d=d, seed=seed)
U = sampler.random(n=N)  # Nx7 em [0,1]
cols = ["Qfeed","Tfeed","cH2S","cNH3","Qreb","Pcolumn","Ecolumn"]

# Limites para teste
# mins = np.array([49500, 119, 2620, 2245, 2.4, 5.9, 69])
# maxs = np.array([50500, 121, 2630, 2255, 3.0, 6.1, 71])

# Limites finais
mins = np.array([47500, 118.95,  875,  750, 2.0, 5.7, 66.5])
maxs = np.array([52500, 125.05, 3500, 3000, 6.0, 6.3, 73.5])

X = mins + U * (maxs - mins)  # Nx7 no espaço físico

# Criando dataframe
df = pd.DataFrame(X, columns=cols)

# Metadados e resultados
df.insert(0, "case_id", np.arange(1, N+1))
df["Tbottom"] = np.nan      # preencher depois
df["Ttop"] = np.nan         # preencher depois

for i, idx in enumerate(np.array_split(df.index, NUM_FILES), start=1):
    path = Path(__file__).resolve().parent / f"sim_cases_{i}.csv"
    df.loc[idx].to_csv(path, index=False)