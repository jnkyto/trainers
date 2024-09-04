# Dataset processing

This package contains various scripts for working on different datasets. **Note:** ORPO-, 
DPO-, and CPO-formatted datasets are the same, so feel free to use these scripts for other 
kinds of trainers too.

### 📁 process_instructional_orpo

Converts instructional datasets to ORPO-format. For Avoin Avustaja and Open Assistant, 
please provide the raw json-export with **label information retained** as the input. In 
a situation where the replies haven't been explicitly ranked, the labels are used to 
calculate a score based on weights, which is then used in sorting. This behavior can 
be overridden with running the processing scripts with `--label_override`, with which 
explicit rankings aren't used at all in favor of labels.

### 📁 sample_and_translate_orpo (for MT fine-tuning)

**Unfinished.** The goal is to sample and translate entries from four different datasets 
sourced from https://opus.nlpl.eu/ and then export everything in ORPO-format.

### 📁 translate_orpo (for MT fine-tuning)

This script is meant for translating an ORPO-formatted dataset in its entirety.