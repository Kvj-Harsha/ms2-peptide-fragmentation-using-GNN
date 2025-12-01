from datasets import load_dataset

ds = load_dataset("bandeiralab/Pep2Prob", split="train")
print(ds[0])
