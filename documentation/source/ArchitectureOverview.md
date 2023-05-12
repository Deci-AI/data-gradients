# Architecture Overview

Data Gradients is an open-source library that provides a set of tools for analyzing and visualizing insights about the dataset.

The process of analysis is divided into next steps:

- Data Ingestion
- Feature Extraction
- Report Generation

## Data Ingestion

Data Ingestion is a process of loading data from the native dataset format into the format suitable for Data Gradients to work on. 
Please see [Dataset Adapters](DatasetAdapters.md) for more details about this step.

## Feature Extraction

Once we have the data in the format suitable for Data Gradients, we can start extracting features from it.
Please see [Features Extraction](FeaturesExtraction.md) for more details about this step.

## Report Generation

After computing all features, what we're left with is a set of aggregate features that contains anonymized 
but highly enriched information about the dataset. The last step of the process is to generate a report
that contains all the insights about the dataset and store or visualize it.
Please see [Reporting](Reporting.md) for more details about this step.
