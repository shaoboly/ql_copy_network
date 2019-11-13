## Summarization

### 0. Prepare

See file headers of `.py` in `preprocess` directory, preprocess each dataset as described.

### 1. Train

Install required library.
 
Run train using **first section** of `template-run.bat` on windows or `template-run.sh` on linux.

On a single 1080Ti, with batch_size = 6, train for 2 epoches takes about 16 hours.

*You may need to change the directory settings and other flags of prepared scripts.*

### 2. Inference

This process generate predicted results and calculate metric of trained model.

Run inference using **second section** of `template-run.bat` on windows or `template-run.sh` on linux.

On a single 1080Ti, with batch_size = 50, beam_size = 4, inference on test set takes about 2 hours.

### 3. Evaluate Metric

Use generated predicted results, this process calculate metric of the results.

Run evaluation using **second section** of `template-run.bat` on windows or `template-run.sh` on linux, add `--eval_only=True` flag.