import json
import numpy as np

with open("results.json", "r") as f:
    results = json.load(f)

c = []
n = []
e = []
cl = []
nl = []
for result in results:
    e.append(results[result][0]["epsilon"])
    c.append(results[result][1]["clean"][0]["predicted"])
    n.append(results[result][2]["noisy"][0]["predicted"])
    cl.append(results[result][1]["clean"][0]["loss"])
    nl.append(results[result][2]["noisy"][0]["loss"])

e = np.array(e)
c = np.array(c)
n = np.array(n)
print("MSE of predicted solution of clean data", np.mean((e - c) ** 2))
print("MSE of predicted solution of noisy data", np.mean((e - n) ** 2))
print("Mean of loss on clean data", np.mean(cl))
print("Mean of loss on noisy data", np.mean(nl))
