# Text Analysis and Metadata Extraction

This Python script analyzes text and extracts metadata such as keywords, sentiment, and content category. It then stores the extracted metadata in a MongoDB collection. The script can read text from a file and perform the analysis on the content.

## Dependencies

- Python 3.6 or higher
- NLTK
- TextBlob
- scikit-learn
- pymongo

To install the required libraries, run:

```bash
pip install nltk textblob scikit-learn pymongo
```

## Usage

To use the script, execute it in the command line with the file path as an argument:

```bash
python meta_extractor.py path/to/text_file.txt
```

The script will read the text file, analyze the text, and store the metadata in a MongoDB collection.

## Features

- Extracts up to 5 most significant keywords from the text using the TF-IDF method
- Determines the sentiment of the text (positive, negative, or neutral)
- Categorizes the content based on predefined training data (e.g., science, technology, sports, politics)
- Stores the extracted metadata in a MongoDB collection

## Document Design

Here is a sample document.

```python
document = {
  "_id": {"$oid":"64517963ea037f029723ebe7"},
  "text":"The new solar panel technology has greatly improved the efficiency of energy conversion.",
  "keywords":
    [
      "panel",
      "technology",
      "efficiency",
      "energy",
      "conversion"
    ],
  "sentiment":"positive",
  "category":"science"
  }
```

## Customization

You can customize the training data for content categorization by modifying the `training_data` variable in the script:

```python
training_data = [
    ('category1', 'Sample text for category 1.'),
    ('category2', 'Sample text for category 2.'),
    ...
]
```

You can also change the maximum number of extracted keywords by modifying the `max_keywords` parameter when calling the `extract_keywords` function:

```python
keywords = extract_keywords(text, max_keywords=5)
```
