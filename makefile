DATA_DIR := data
GRAPE_DIR := $(DATA_DIR)/GRAPE

# GRAPE dataset: https://springernature.figshare.com/collections/_/6406319
# Mapping: Figshare file ID : local filename
FIGSHARE_FILES := \
	41670009:VFs_and_clinical_info.xlsx \
	41358156:CFPs.rar \
	41358159:annotated_images.rar \
	41358162:annotations_json.rar \
	41358150:ROI_images.rar

all: data

data:
	@mkdir -p $(GRAPE_DIR)
	@echo "Downloading GRAPE Figshare files..."
	@for file in $(FIGSHARE_FILES); do \
		ID=$${file%%:*}; \
		NAME=$${file##*:}; \
		if [ ! -f $(GRAPE_DIR)/$$NAME ]; then \
			echo "Downloading $$NAME from Figshare file ID $$ID..."; \
			wget -O $(GRAPE_DIR)/$$NAME "https://api.figshare.com/v2/file/download/$$ID"; \
		else \
			echo "$$NAME already exists, skipping."; \
		fi; \
	done

clean-data:
	rm -rf $(DATA_DIR)
	@echo "Data directory removed."
