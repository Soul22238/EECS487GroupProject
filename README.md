# EECS 487 Group18 Evaluating Prompt Engineering Strategies for LLM-Based Code Generation and Debugging

## HumanEval: Hand-Written Evaluation Set 
The dataset and codes are largely adapted from the paper "[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)".
### Citation

Please cite their work using the following bibtex entry:

```
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
### 
Make sure that you are in the DS-1000-dataset path. If not, you can
```shell
cd .../DS-1000-dataset 
```

### Installation

Make sure to use python 3.7 or later:
```
$ conda create -n codex python=3.7
$ conda activate codex
```

Check out and install this repository:
```
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```
### Preprocessing and Experiments
The preprocessing and experiements to get the responses are in human-eval/Experiments/generate_codes.py and human-eval/Experiments/meta_prompts.py.
Before running the python files, make sure to replace the API key in the config file with your own.


### Evaluation and Results

All the results are saved in the human-eval/Experiments/Results folder.


Please make sure all the answer files are placed in the data folder.
Check whether your current working directory is the human-eval folder:
```shell
pwd
.../human-eval  # your actual path may differ
```
Then run the following command:
```shell
evaluate_functional_correctness data/gemini-1.5-flash-Instruction2CodesCoT-164-answers.jsonl --problem_file=data/HumanEval.jsonl 
```

The return should look like the following:
evaluate_functional_correctness data/gemini-1.5-flash-Instruction2CodesCoT-164-answers.jsonl --problem_file=data/HumanEval.jsonl      
Reading samples...
492it [00:00, 37558.20it/s]
Running test suites...
100%|███████████████████████████████████████████████████████████| 492/492 [00:04<00:00, 117.12it/s]
Writing results to data/gemini-1.5-flash-Instruction2CodesCoT-164-answers.jsonl_results.jsonl...
100%|█████████████████████████████████████████████████████████| 492/492 [00:00<00:00, 98954.52it/s]
{'pass@1': 0.8130081300813007}


The result file will be saved in the human-eval/data folder.

To compute the accuracy, open human-eval/Experiments/get_accuracy.py and update the path to the result file accordingly.

Finally, run the script to see the evaluation result:
```shell
python3 human-eval/Experiments/get_accuracy.py
```




## DS-1000
Make sure that you are in the DS-1000-dataset path. If not, you can
```shell
cd .../HumanEval-dataset 
```
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

