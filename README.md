# EECS 487 Group18 Evaluating Prompt Engineering Strategies for LLM-Based Code Generation and Debugging
## DS-1000
### Declaration
The codes and data are largely adapted from the paper [_DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation_](https://arxiv.org/abs/2211.11501). The oriinal github repository is attached here [project page](https://ds1000-code-gen.github.io/) .
### Citation for the original work

```
@article{Lai2022DS1000,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Yuhang Lai and Chengxi Li and Yiming Wang and Tianyi Zhang and Ruiqi Zhong and Luke Zettlemoyer and Scott Wen-tau Yih and Daniel Fried and Sida Wang and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.11501}
}
```
### Preprocessing and Experiments
* Preprocessing: 
We used Trimming.py along with the configuration file trim_config.json to clean and summarize the original responses.
Before running the script, make sure to replace the API key in the config file with your own.
Then, execute the following command to generate the trimmed results:
```shell
python Trimming.py
```
The trimmed prompts are stored in ./DS-1000/data/trimmed_prompts.

* Experiments: 
Open ModelAPI/config.json and replace the placeholder "your_key" with your actual API key.
Then 
```shell
python ModelAPI/generate_codes.py
```
During this process, the model will save the raw responses returned by the API in the ./ModelAPI/RawCodes/model_name directory.

### Evaluation and Results

```shell

python test_ds1000.py --model gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-answers
```
A gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-result.txt file should be generated in the results folder.
All the summarized accuracy data can be found within the results folder.

