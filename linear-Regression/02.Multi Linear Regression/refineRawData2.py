import csv

data_keys = ['index','average annual preceipitation',
			'average January temp','average July temp',
			'1960 SMSA','household size','schooling for over 22',
			'household with full kitchens','population per square mile',
			'nonwhite population','office workers','poor families',
			'pollution potential of hydrocarbons',' pollution potential of oxides of Nitrogen',
			'pollution of Sulfur Dioxide','relative humidity','death rate']
data_dict = {p:'' for p in data_keys}
with open('Data/data.csv','w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames = data_keys)
	writer.writeheader()
	with open('Data/raw_data/raw_data.txt') as f:
		for lines in f:
			data = [d for d in lines.replace('\n','').split(' ') if d != '']
			for j in range(len(data_keys)):
				data_dict[data_keys[j]]= data[j]
			writer.writerow(data_dict)
