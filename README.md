# Multimodal Puzzle Reasoning


## Algorithmic Puzzles

We introduce the novel task of multimodal puzzle solving, framed within the context of visual question-answering. We present a new dataset, AlgoPuzzleVQA designed to challenge and evaluate the capabilities of multimodal language models in solving algorithmic puzzles that necessitate both visual understanding, language understanding, and complex algorithmic reasoning. We create the puzzles to encompass a diverse array of mathematical and algorithmic topics such as boolean logic, combinatorics, graph theory, optimization, search, etc., aiming to evaluate the gap between visual data interpretation and algorithmic problem-solving skills. The dataset is generated automatically from code authored by humans. All our puzzles have exact solutions that can be found from the algorithm without tedious human calculations. It ensures that our dataset can be scaled up arbitrarily in terms of reasoning complexity and dataset size. Our investigation reveals that large language models (LLMs) such as GPT4V and Gemini exhibit limited performance in puzzle-solving tasks. We find that their performance is near random in a multi-choice question-answering setup for a significant number of puzzles. The findings emphasize the challenges of integrating visual, language, and algorithmic knowledge for solving complex reasoning problems.


## Visual Features of the Puzzles

The configuration of the puzzle/problem is shown as an image, which constitutes its visual context. We identify the following fundamental aspects of the visual context that influence the nature of the puzzles:

 - Colour
 - Position
 - Shape/Size
 - Text

<p align="center">
  <img src=img/teaser_visual.png />
</p>


## Algorithmic Features of the Puzzles

We also identify the algorithmic concepts required for solving the puzzles i.e. for answering the questions for the puzzle instances. They are as follows:

 - Arithmetic
 - Boolean Logic
 - Combinatorics
 - Graphs
 - Optimization
 - Search
 - Sets

<p align="center">
  <img src=img/teaser_algorithm.png />
</p>

The algorithmic categories are not mutually exclusive, as we need to use two or more categories to derive the answer for most puzzles.


## Dataset

The dataset is available in [here](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA/data). We created a total of 18 different puzzles spanning various algorithmic and mathematical topics. Many of these puzzles are popular in various recreational or academic settings.

In total, we have 1800 instances from the 18 different puzzles. These instances are analogous to different test cases of the puzzle, i.e.  they have different input combinations, initial and goal states, etc. Reliably solving all the instances would require finding the exact algorithm to use and then applying it accurately. This is akin to how we verify the accuracy of a computer program aiming to solve a particular task through a broad range of test cases.

We currently consider the full dataset as an evaluation-only benchmark. The detailed examples of all puzzles are shown [here](https://github.com/declare-lab/puzzle-reasoning/blob/master/puzzles.md).

The code for generating the dataset can be found [here](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA/generation). The number of instances and the difficulty of the puzzles can be scaled arbitrarily to any desired size or level.


## Ontology

The ontological categorization of the puzzles are as follows:

<p align="center">
  <img src=img/ontology.png />
</p>


## Experiments

The experimental setup and scripts can be found in the [AlgoPuzzleVQA](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA) directory.


## Citation

Please consider citing the following article if you found our work useful:

```bibtex
@article{ghosal2024algopuzzlevqa,
  title={Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning},
  author={Ghosal, Deepanway and Han, Vernon Toh Yan and Chia, Yew Ken and and Poria, Soujanya},
  journal={arXiv preprint arXiv:2403.?????},
  year={2024}
}
```