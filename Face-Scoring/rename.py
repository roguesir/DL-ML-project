import os       
import xlrd

data = xlrd.open_workbook('./Rating_Collection/Attractiveness-label.xlsx')
path = './web_image' 
data = data.sheet_by_index(0)   
col_score = data.col_values(1)

#print(round(col_score[1],2))
for i in range(0,500):
	score = str(round(col_score[i])*2)
	for image in os.listdir(path):
		#print(image)
		if os.path.isfile(os.path.join(path, image)):
			id_tag = image.find("j")
			score_tag = image.find("-")
			#print(id_tag)
			name=image[score_tag+1:id_tag-1]
			#print(name)
			#print(i+1)
			if name == str(i+1):
				os.rename(path + os.sep + image, path + os.sep + score + '-' + name + '.jpg')