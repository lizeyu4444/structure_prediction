import codecs

data = []
with codecs.open('test.txt', 'r') as fi:
    for i in fi.readlines():
    	data.append([j for j in i.split() if j.strip()])


print(data[0])
print(data[1])
print(data[2])
# print(data)