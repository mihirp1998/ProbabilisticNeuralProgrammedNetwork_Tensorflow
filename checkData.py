import glob
import pickle
a = glob.glob("data/CLEVR/CLEVR_64_MULTI_LARGE/trees/train/*")
b = glob.glob("data/CLEVR/CLEVR_64_MULTI_LARGE/images/train/*")
a.sort()
b.sort()
#print(len(zip(a,b)))
print(len(a),len(b))
im  = list(zip(a,b))
print(im[0])
filter_im = []
filter_tree = []
for i in range(len(a)):
	p = pickle.load(open(im[i][0],"rb"))
	if p.function == "describe":
		filter_im.append(im[i][1])
		filter_tree.append(im[i][0])
	#m = [i for i in p if i.function == "describe"]
print(len(filter_im),len(filter_tree))
print(filter_im[:2],filter_tree[:2])
pickle.dump(filter_im,open("data/filter_im.p","wb"))
pickle.dump(filter_tree,open("data/filter_tree.p","wb"))
