# devnagri dataset
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00389/DevanagariHandwrittenCharacterDataset.zip
! unzip DevanagariHandwrittenCharacterDataset.zip -d data
path = "/content/data/DevanagariHandwrittenCharacterDataset"
data = ImageDataBunch.from_folder(path, train='Train',test='Test',valid_pct=0.2, ds_tfms = tfms , size= 26)

data.show_batch(2, figsize= (7,6))
learn = cnn_learner(data, models.resnet50, metrics= error_rate)
learn.fit_one_cycle(5)

# epoch	train_loss	valid_loss	error_rate	time
# 0	2.200161	1.468215	0.419054	02:36
# 1	0.996351	0.527132	0.157928	02:34
# 2	0.550935	0.254312	0.077877	02:33
# 3	0.398611	0.178680	0.054476	02:34
# 4	0.359773	0.179330	0.053261	02:33

learn.save("model_devnagri_v1")

learn.lr_find()
learn.recorder.plot()

learn.unfreeze()  # unfreeze model and train on different lr to check model performance
learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))
