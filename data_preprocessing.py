import pandas as pd

def main():
  
    preprocess(pd.read_csv(f"./metadata/{filenames[0]}"))

filenames = ["LegalEagle_metadata.csv", 
             "Mark Rober_metadata.csv", 
             "Matt D'Avella_metadata.csv", 
             "MrBeast_metadata.csv", 
             "NikkieTutorials_metadata.csv",
             "Peter McKinnon_metadata.csv",
             "Rachel & Jun's Adventures!_metadata.csv",
             "The Slow Mo Guys_metadata.csv",
             "Veritasium_metadata.csv"
             "Yes Theory_metadata.csv"
             ]

# for filename in filenames:
#     pd.read_csv(filename)

def preprocess(data: pd.DataFrame):
    answer = data.head(20)
    answer.append(data.tail(20), ignore_index = True, inplace = True)

if __name__ == "__main__":
    main()  
