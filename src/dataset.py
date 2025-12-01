from datasets import load_dataset

def load_pep2prob(split="train"):
    dataset = load_dataset("bandeiralab/Pep2Prob", split=split)
    return dataset
