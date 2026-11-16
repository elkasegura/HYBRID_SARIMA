#Import the file to read it
import pandas as pd
disocupati_trimestral = pd.read_csv('/home/lka/HYBRID_SARIMA/data.csv', 
                       index_col='Period', 
                       parse_dates=True)
print(disocupati_trimestral)