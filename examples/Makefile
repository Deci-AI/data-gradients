# ** Commands to run locally the notebooks **
# This is exclusively meant for testing and debugging quickly.
# /!\ Any inline shell command (e.g. `!pip install ...`) is ignored, so make sure to have the right environment locally

DETECTION_ROBOFLOW = detection_from_roboflow.ipynb

SEGMENTATION_COCO = segmentation_tinycoco.ipynb
SEGMENTATION_VOC = segmentation_voc.ipynb

CLASSIFICATION_CALTECH100 = classification_torchvision_caltech101.ipynb
CLASSIFICATION_FASHION_MNIST = classification_torchvision_fashion_mnist.ipynb


run_detection_from_roboflow:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(DETECTION_ROBOFLOW) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

run_segmentation_coco:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(SEGMENTATION_COCO) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

run_segmentation_voc:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(SEGMENTATION_VOC) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

run_classification_caltech101:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(CLASSIFICATION_CALTECH100) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

# FIXME
run_classification_fashion_mnist:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(CLASSIFICATION_FASHION_MNIST) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR
