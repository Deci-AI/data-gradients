ROBOFLOW_NOTEBOOK = examples/detection_from_roboflow.ipynb
CALTECH100_NOTEBOOK = examples/classification_torchvision_caltech101.ipynb
SEG_COCO_NOTEBOOK = examples/segmentation_tinycoco.ipynb

# Make sure the notebooks work properly

run_detection_from_roboflow:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(ROBOFLOW_NOTEBOOK) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

run_seg_coco:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(SEG_COCO_NOTEBOOK) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR

run_caltech101:
	TMP_DIR=$$(mktemp -d) ; \
	jupyter nbconvert \
	    --to python \
	    --RegexRemovePreprocessor.patterns="^\!" $(CALTECH100_NOTEBOOK) \
	    --output-dir=$$TMP_DIR --output=_notebook.py ; \
	python $$TMP_DIR/_notebook.py ; \
	rm -rf $$TMP_DIR
