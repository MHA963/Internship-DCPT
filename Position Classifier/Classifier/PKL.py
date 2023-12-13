import pandas as pd

data_frame = pd.read_pickle("C:/Users/student/Desktop/images/merged_file_for_testing_2.pkl")

print("")


# pkl syntax (case sensitive)
# calling the data frame
# data_frame will spit out the pkl file with everything in it 

# data_frame['year'] will spit out all the files in year order

# data_frame['Institution'] will spits out all the files witht the relative institution

# for col in data_frame.columns: 
# print(col) will spits out all the columns in the pkl file

# data_frame[['Institution', 'Year', 'repeated path', 'Randomization number']] 
# will spit out a table with the institution, yearm repeated path and randomization number 

# data_frame = data_frame[['Institution', 'Year', 'repeated path', 'Randomization number']]
# will minimize the data file to a table with the named parameter

#data_frame[data_frame['Institution'] == 'Aalborg'] spits out the table for the specified instituation

#data_frame['Institution'].unique() will spit out the unique instituation names in the pkl file

#data_frame['repeated path'].iloc[0] will spit out the first index in the reapeated path column
