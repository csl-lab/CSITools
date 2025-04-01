## ----------------------------------------------------------------------------
## Imports

import scipy.io as scio
import numpy as np
import h5py

import sys

if "../../.." not in sys.path:
    sys.path.insert(0, "../../..")

def loadTestData(usamp : int  			 = 1,
								 nte   : int | float = np.inf):
	# filename
	fname = 'newPhantom12EchoesMPR2'
	# source folder
	folder = '../../data/'
	# load and return
	S, te, field, xGT = loadPhantomSimu(folder, fname)
	c = xGT[::usamp, ::usamp, :2]
	x = xGT[::usamp, ::usamp,  2]
	if S.shape[2] < nte:
		return S[::usamp,::usamp, :], te, field, c, x
	return S[::usamp,::usamp, :nte], te[:nte], field, c, x

def loadDataEchoes(path, datstr):
	try:
		data = scio.loadmat(path + datstr + '.mat')
		So = data['EchoCoilImages']
		print(So.shape)
		sshapeo     = So.shape
		teo         = np.unique(data['TE'].ravel())
		nteo        = len(teo)
		field       = 1.5
		print('EchoCoilImages format', teo)
	except:
		data        = scio.loadmat(path + datstr + '.mat')
		So        = data['imDataParams'][0, 0]['images']
		So          = So[:,:,:,:,:]
		print(So.shape)
		sshapeo     = So.shape
		teo         = np.unique(data['imDataParams'][0, 0]['TE'].ravel())
		nteo        = len(teo)
		field       = data['imDataParams'][0, 0]['FieldStrength'][0]
		print('imDataParams format')

	S = So
	te = teo.ravel()

	return S, te, field

def loadMrfData5(path, datstr):
	try:
		data = scio.loadmat(path + datstr + '.mat')
		So = data['EchoCoilImages']
		print(So.shape)
		sshapeo     = So.shape
		teo         = data['TE'].ravel()
		nteo        = len(teo)
		field       = 1.5
		print('EchoCoilImages format', teo)
	except:
		data        = scio.loadmat(path + datstr + '.mat')
		So        = data['imDataParams'][0, 0]['images']
		So          = So[:,:,:,:,:]
		print(So.shape)
		sshapeo     = So.shape
		teo         = data['imDataParams'][0, 0]['TE'].ravel()
		nteo        = len(teo)
		field       = data['imDataParams'][0, 0]['FieldStrength'][0]
		print('imDataParams format')

	S = So
	te = teo.ravel()

	return S, te, field

def loadPhantomSimu(path, datstr):
	data = scio.loadmat(path + datstr + '.mat')
	So = data['imDataParams'][0, 0]['images']
	So = So[:,:,:,:,:]
	# print(So.shape)
	sshapeo = So.shape
	teo = np.unique(data['imDataParams'][0, 0]['TE'].ravel())
	# nteo = len(teo)
	field = data['imDataParams'][0, 0]['FieldStrength'][0]
	# print('imDataParams format')

	S = So.squeeze()
	te = teo.ravel()
	xGT = data['xGT']
	
	return S, te, float(field[0]), xGT