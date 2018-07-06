import string
import csv
from random import *

data_keys = {'name','special_customer','annual_income','money_spent_in_store'}

data_dict = {p:'' for p in data_keys}

def generate_name():
	min_char = 8
	max_char = 12

	character = string.ascii_letters + string.punctuation + string.digits
	name = "".join(choice(character) for x in range(randint(min_char,max_char)))
	return name

def generate_yes_no():
	return randint(0,1)

def generate_salary():
	return randint(15000,100000)

def money_on_grocery():
	return randint(10,1200)

with open('data.csv','w') as csv_file:
	writer = csv.DictWriter(csv_file,fieldnames = data_keys)
	writer.writeheader()
	for i in range(1,1000):
		data_dict['name'] = generate_name()
		data_dict['special_customer'] = generate_yes_no()
		data_dict['annual_income'] = generate_salary()
		data_dict['money_spent_in_store'] = money_on_grocery()
		writer.writerow(data_dict)