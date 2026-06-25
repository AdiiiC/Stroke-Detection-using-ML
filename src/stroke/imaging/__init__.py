"""Imaging branch: CT/MRI deep learning for stroke detection.

Modules
-------
* ``dataset``  -- torch datasets/dataloaders (ImageFolder + synthetic)
* ``model``    -- transfer-learning CNN classifier
* ``unet``     -- U-Net for ischemic-lesion segmentation
* ``gradcam``  -- Grad-CAM saliency for the classifier
* ``train_cnn``-- training loop with the --quick smoke-test mode
* ``fusion``   -- late fusion of tabular risk + imaging probability
"""
