# Table 5 — OSTQuant and SCNA model quality

| Model | Quantization | Method | Paper PPL | Actual PPL | Paper Acc. (%) | Actual Acc. (%) | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2_7b | W6A6 | OSTQuant | 5.5 | 5.50187873840332 | 65.09 | 65.09 | True |
| llama3_8b | W6A6 | OSTQuant | 6.23 | 6.229605674743652 | 68.03 | 68.03 | True |
| llama2_7b | W6A6 | SCNA-8 | 5.54 | 5.535458087921143 | 64.99 | 64.99000000000001 | True |
| llama3_8b | W6A6 | SCNA-8 | 6.32 | 6.315341472625732 | 68.01 | 68.01 | True |
| llama2_7b | W6A6 | SCNA-16 | 5.5 | 5.5032877922058105 | 65.04 | 65.03999999999999 | True |
| llama3_8b | W6A6 | SCNA-16 | 6.24 | 6.237985134124756 | 68.02 | 68.02 | True |
| llama2_7b | W6A6 | SCNA-32 | 5.5 | 5.504350185394287 | 65.06 | 65.06 | True |
| llama3_8b | W6A6 | SCNA-32 | 6.24 | 6.237723350524902 | 67.97 | 67.97 | True |
| llama2_7b | W4A4 | OSTQuant | 5.94 | 5.946303844451904 | 62.58 | 62.580000000000005 | True |
| llama3_8b | W4A4 | OSTQuant | 7.33 | 7.32977819442749 | 64.64 | 64.64 | True |
| llama2_7b | W4A4 | SCNA-8 | 6.0 | 5.9992594718933105 | 62.44 | 62.44 | True |
| llama3_8b | W4A4 | SCNA-8 | 7.44 | 7.444907188415527 | 64.44 | 64.44 | True |
| llama2_7b | W4A4 | SCNA-16 | 5.96 | 5.957137107849121 | 62.93 | 62.93 | True |
| llama3_8b | W4A4 | SCNA-16 | 7.34 | 7.344878673553467 | 65.11 | 65.11 | True |
| llama2_7b | W4A4 | SCNA-32 | 5.95 | 5.95246696472168 | 62.77 | 62.77 | True |
| llama3_8b | W4A4 | SCNA-32 | 7.35 | 7.346691131591797 | 64.72 | 64.72 | True |
