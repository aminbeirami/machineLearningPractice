import csv

data_keys = ['index','brain_weight','body_weight']
data_dict = {p:'' for p in data_keys}
with open('Data/data.csv','w') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames = data_keys)
	writer.writeheader()
	with open('Data/raw_data/data.txt') as f:
		for items in f:
			data = [d for d in items.replace('\n', '').split(' ') if d != '']
			for j in range(len(data_keys)):
				data_dict[data_keys[j]] = data[j]
			writer.writerow(data_dict)