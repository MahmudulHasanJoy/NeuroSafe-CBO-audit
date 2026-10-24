import pandas as pd
d = "results/v3/ihdp/"
a2 = pd.read_csv(d + "ablation2_episodes.csv")
keep = ["Ctx-noGLM", "Ctx-noBucket", "Ctx-noKalman", "Ctx-noContext"]
a2[a2.agent.isin(keep)].to_csv(d + "ablation_episodes.csv", index=False)
a2[a2.agent == "Ctx-BetaCell"].to_csv(d + "posthoc_betacell_episodes.csv", index=False)
print("IHDP ablation rows per agent:", a2[a2.agent.isin(keep)].groupby("agent").size().to_dict())
for dom in ["scm", "lending"]:
    ab = pd.read_csv(f"results/v3/{dom}/ablation_episodes.csv")
    print(f"{dom} ablation agents: {sorted(ab.agent.unique())}")
