# Socratic Debugging Benchmark

![Socrates and a student](annotation_tool/img/socratic_debugging.png)

The repository contains the code and dataset for accompanying the paper [Socratic Questioning of Novice Debuggers: A Benchmark Dataset and Preliminary Evaluations"](https://aclanthology.org/2023.bea-1.57/) in proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Application at ACL 2023.

There are 3 subfolders:
* evaluation_dataset: Contains all the dialogues used for evaluations in the paper. This subfolder excludes any dialogues with analogies that were not used for evaluations.
* final_dataset: Contains all 86 dialogues curated for the dataset. This subfolder includes dialogues with analogy use.
* manual_evaluation_dialogues: Contains the dialogues used for manual evaluation of the dataset.

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

## Citation
If you use this dataset, please cite our [paper](https://aclanthology.org/2023.bea-1.57/):
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
