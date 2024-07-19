# Processing data exported from Open Assistant

These scripts process Open Assistant data to the format expected by CPO-like trainers,
where the prompt is followed by chosen and rejected replies. More information can be found 
at https://huggingface.co/docs/trl/main/en/cpo_trainer.

### Note: These scripts require the user to provide data with label information retained!  
In a situation where the replies haven't been explicitly ranked, the labels are used to 
calculate a score based on weights, which is then used in sorting. This behavior can 
be overridden with running the processing scripts with `--label_override`, with which 
explicit rankings aren't used at all in favor of labels.