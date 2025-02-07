import json
import numpy as np

with open("results.json", "r") as f:
    results = json.load(f)

cmse = []
nmse = []
cl = []
nl = []

for result in results:
    cmse.append(results[result][1]["clean"][0]["mse"])
    nmse.append(results[result][2]["noisy"][0]["mse"])
    cl.append(results[result][1]["clean"][0]["loss"])
    nl.append(results[result][2]["noisy"][0]["loss"])

cmse = np.array(cmse)
nmse = np.array(nmse)
cl = np.array(cl)
nl = np.array(nl)
print("Average MSE clean: ", np.mean(cmse))
print("Average MSE noisy: ", np.mean(nmse))
print("Average Loss clean: ", np.mean(cl))
print("Average Loss noisy: ", np.mean(nl))