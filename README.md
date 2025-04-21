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
* Preprocessing 
We used Trimming.py and trim_config.json to trim and summarize the original responses. 
You will need to replace the api key with your key and run the following command to get the results.
```shell
python Trimming.py
```
The trimmed prompts are stored in ./DS-1000/data/trimmed_prompts.

* Experiments
First revise the ModelAPI/config.jsonModelAPI/config.json.
Replace your_key with your API key.
Then 
```shell
python ModelAPI/generate_codes.py
```
In this process, the model will record the responses returned by the API (raw responses)

### Evaluation and Results

```shell

python test_ds1000.py --model gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-answers
```
There should be a generated gemini-1.5-flash-TrimmedInstruction2CodesBase-1000-result.txt in the results folder.
All the summarized accuracies are in the results folder.

