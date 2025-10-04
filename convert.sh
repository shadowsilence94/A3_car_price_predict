#!/bin/bash

# This script converts a Jupyter Notebook to a PDF.
# It first clears the output from the notebook to avoid errors
# with special characters, and then it uses nbconvert to
# create the PDF.

# The name of the notebook to convert.
NOTEBOOK_FILE="A3_car_price_classification.ipynb"

# The name of the output PDF file.
OUTPUT_FILE="A3.pdf"

# Run nbconvert with the ClearOutputPreprocessor.
jupyter nbconvert --to pdf --ClearOutputPreprocessor.enabled=True "$NOTEBOOK_FILE" --output "$OUTPUT_FILE"

# Check if the conversion was successful.
if [ $? -eq 0 ]; then
    echo "Successfully converted $NOTEBOOK_FILE to $OUTPUT_FILE"
else
    echo "Error converting $NOTEBOOK_FILE to $OUTPUT_FILE"
fi
