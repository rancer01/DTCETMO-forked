import torch
import torchvision
import torch.optim
import os
import model
import numpy as np
import glob
import time
import imageio
import cv2


def dataloader(path):
	data_hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	shp1,shp2,shp3 = data_hdr.shape
	shp =  tuple([shp2,shp1])
	data_hdr = cv2.resize(data_hdr, (1024, 1024))
	data_hdr = torch.from_numpy(data_hdr).float()
	data_hdr = data_hdr.permute(2, 0, 1)
	data_hdr = data_hdr.cuda().unsqueeze(0)
	return data_hdr,shp


def hdr2sdr(image_path):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	scale_factor = 64
	tmo_net = model.Tmonet(scale_factor).cuda()
	tmo_net = torch.nn.DataParallel(tmo_net).cuda()
	tmo_net.load_state_dict(torch.load('Trial-16/Epoch499.pth'))
	img_hdr, shp = dataloader(image_path)
	start = time.time()
	xu,lumii,img_ldr,out2561,out2562,inp = tmo_net(img_hdr)
	end_time = (time.time() - start)
	print(end_time)
	os.makedirs('data/L1', exist_ok=True)
	os.makedirs('data/large', exist_ok=True)
	image_path = image_path.replace('comb_hdrf','L1')
	image_path = image_path.replace('hdr', 'jpg')
	# im2_lay1 = image_path.replace('dummy','large',)
	result_path = image_path
##
	# numi = im2_lay1.strip('data/large').strip('.bmp')
	# numi = numi[1:]
	# print(numi)
	o1 = out2561.squeeze().flatten().tolist()
	o2 = out2562.squeeze().flatten().tolist()
	ip = inp.squeeze().flatten().tolist()
	lumiil = lumii.squeeze().flatten().tolist()
	xil = xu.squeeze().flatten().tolist()
	# x_dee = x_d.squeeze().flatten().tolist()
	lay11out_h = [hex(np.float16(value).view('H'))[2:].zfill(4) for value in o1]
	lay11in_h = [hex(np.float16(value).view('H'))[2:].zfill(4) for value in o2]
	lay11oin_h = [hex(np.float16(value).view('H'))[2:].zfill(4) for value in ip]
	lumiil_h = [hex(np.float16(value).view('H'))[2:].zfill(4) for value in lumiil]
	xil_h = [hex(np.float16(value).view('H'))[2:].zfill(4) for value in xil]
	
	with open('out61.txt', 'w') as file:
		for value in lay11out_h:
			file.write(str(value) + '\n')
	with open('out62.txt', 'w') as file:
		for hex_value in lay11in_h:
			file.write(hex_value + '\n')
	with open('inp.txt', 'w') as file:
		for hex_value in lay11oin_h:
			file.write(hex_value + '\n')
	with open('lumii.txt', 'w') as file:
		for hex_value in lumiil_h:
			file.write(hex_value + '\n')
	with open('xii.txt', 'w') as file:
		for hex_value in xil_h:
			file.write(hex_value + '\n')

# ##	
# 	torchvision.utils.save_image(lay1, im2_lay1)
	torchvision.utils.save_image(img_ldr, result_path)
	image = cv2.imread(result_path)
	swapped_image = image.copy()
	swapped_image[:, :, 0], swapped_image[:, :, 2] = image[:, :, 2], image[:, :, 0]
	#swapped_image = cv2.resize(swapped_image, shp)
	cv2.imwrite(result_path, swapped_image)



if __name__ == '__main__':
	with torch.no_grad():
		filePath = 'data/comb_hdrf'
		test_list = glob.glob(filePath+"/*hdr")
		image = 'Test.hdr'
		print(image)
		hdr2sdr(image)
		#for image in test_list:
		#	print(image)
		#	hdr2sdr(image)
