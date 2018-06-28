#This file creates a CSV file out of raw data
import csv

#Create CSV file with appropriate header
data_keys = ['vendor name','model_name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
data_dict = {p:'' for p in data_keys}
with open('Data/data.csv','w') as csvfile:
	writer = csv.DictWriter(csvfile,fieldnames = data_keys)
	writer.writeheader()
	# open the raw_file
	with open('Data/raw_data/Data.data') as f:
		for items in f:
			# split the data values and remove unnecessary characters
			line = items.replace('\n','').split(',')
			for j in range(len(data_keys)):
				data_dict[data_keys[j]]=line[j]
			#write the splitted data values to the csv file
			writer.writerow(data_dict)
