# Socratic Debugging Benchmark

<p align="center">
  <img src="annotation_tool/img/socratic_debugging.png" height="400">
</p>

The repository contains the source code, data contribution tool, and dataset for the Socratic Debugging Benchmark accompanying the papers ["Socratic Questioning of Novice Debuggers: A Benchmark Dataset and Preliminary Evaluations"](https://aclanthology.org/2023.bea-1.57/) in proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Application at ACL 2023 and ["Can Language Models Employ the Socratic Method? Experiments with Code Debugging"](https://dl.acm.org/doi/10.1145/3626252.3630799) in the proceedings of SIGCSE'24.

There are 2 versions of the dataset available in this repository version 1 was submitted to BEA@ACL'24 and version 2 was submitted to SIGCSE'24. The main difference between the two versions is that version 2 contains more dialogues. The dataset is available in the `final_dataset` folder. The `evaluation_dataset` folder contains the dialogues used for evaluations in the paper. The only difference between the `final_dataset` and `evaluation_dataset` is that the `evaluation_dataset` excludes any dialogues with analogies that were not used for evaluations. The `manual_evaluation_dialogues` folder contains the dialogues used for manual evaluation of the dataset. `train` was used in the fine-tuning experiments while `test` was used for automatically evaluating the fine-tuned model.

If you have any questions please contact [ealhossa@uncc.edu](mailto:ealhossa@uncc.edu).

## Running the Data Contribution Tool
Create a new virtual Python environment called `socratic_env` using Anaconda or virtualenv. Then, activate the environment and install the required packages using the following command:
```
pip install -r requirements.txt
```
To run the tool, run the following command:
```
streamlit run annotation_tool\🏡Getting_Started.py
```
The tool will open in a new browser window. 

The bugs used in this dataset along with some metadata can be found in `annotation_tool\bug_repo.xlsx`. The tool will automatically load the bugs from this file. If you want to add new bugs, you can add them to this file. The tool will automatically load the new bugs the next time it is run.

## Running Experiments

Add your open ai key into the following file:
```
.streamlit\oai_key.txt
```

Then, run baseline experiments using the following command:
```
bash launch_runs.sh
```
The results will be saved in the `results` folder.

## TODOs:
- [x] Add version 2 of the dataset
- [x] Add code for fine-tuning FLAN-T5.
- [x] Add code for CoT experiments.
- [ ] Add the benchmark and metrics to huggingface.
- [x] Update with SIGCSE paper link and citation.
- [ ] Add version 3 of the dataset
- [ ] Add LM outputs
## Citation
If you use this dataset, please cite our papers: 
* [BEA@ACL'23](https://aclanthology.org/2023.bea-1.57/)
* [SIGCSE'24](https://dl.acm.org/doi/10.1145/3626252.3630799)
```
@inproceedings{al-hossami-etal-2023-socratic,
    title = "Socratic Questioning of Novice Debuggers: A Benchmark Dataset and Preliminary Evaluations",
    author = "Al-Hossami, Erfan  and
      Bunescu, Razvan  and
      Teehan, Ryan  and
      Powell, Laurel  and
      Mahajan, Khyati  and
      Dorodchi, Mohsen",
    booktitle = "Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bea-1.57",
    pages = "709--726",
    abstract = "Socratic questioning is a teaching strategy where the student is guided towards solving a problem on their own, instead of being given the solution directly. In this paper, we introduce a dataset of Socratic conversations where an instructor helps a novice programmer fix buggy solutions to simple computational problems. The dataset is then used for benchmarking the Socratic debugging abilities of GPT-based language models. While GPT-4 is observed to perform much better than GPT-3.5, its precision, and recall still fall short of human expert abilities, motivating further work in this area.",
}
```

```
@inproceedings{al-hossami-etal-2024-can,
author = {Al-Hossami, Erfan and Bunescu, Razvan and Smith, Justin and Teehan, Ryan},
title = {Can Language Models Employ the Socratic Method? Experiments with Code Debugging},
year = {2024},
isbn = {9798400704239},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626252.3630799},
doi = {10.1145/3626252.3630799},
abstract = {When employing the Socratic method of teaching, instructors guide students toward solving a problem on their own rather than providing the solution directly. While this strategy can substantially improve learning outcomes, it is usually time-consuming and cognitively demanding. Automated Socratic conversational agents can augment human instruction and provide the necessary scale, however their development is hampered by the lack of suitable data for training and evaluation. In this paper, we introduce a manually created dataset of multi-turn Socratic advice that is aimed at helping a novice programmer fix buggy solutions to simple computational problems. The dataset is then used for benchmarking the Socratic debugging abilities of a number of language models, ranging from fine-tuning the instruction-based text-to-text transformer Flan-T5 to zero-shot and chain of thought prompting of the much larger GPT-4. The code and datasets are made freely available for research at the link below.},
booktitle = {Proceedings of the 55th ACM Technical Symposium on Computer Science Education V. 1},
pages = {53–59},
numpages = {7},
keywords = {benchmark dataset, debugging, language models, socratic dialogue},
location = {<conf-loc>, <city>Portland</city>, <state>OR</state>, <country>USA</country>, </conf-loc>},
series = {SIGCSE 2024}
}

```
