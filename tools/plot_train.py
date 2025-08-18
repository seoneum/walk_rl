import sys, pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1])
df.columns = ["update","avgR","doneEps","fps"]
fig,ax = plt.subplots(1,2, figsize=(10,4))
df.plot(x="update", y="avgR", ax=ax[0]); ax[0].grid(True)
df.plot(x="update", y="fps", ax=ax[1]); ax[1].grid(True)
plt.tight_layout(); plt.show()
