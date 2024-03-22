# Multimodal Puzzle Reasoning with LLMs

> 🔥 PuzzleVQA, our new dataset reveals serious challenges of multimodal LLMs in understanding simple abstract patterns.
[Paper](https://arxiv.org/abs/2403.13315) | [Website](https://puzzlevqa.github.io/)

> 📣 We are releasing AlgoPuzzleVQA, a novel and challenging dataset for multimodal reasoning! Soon, we will release more multimodal puzzle datasets. Stay tuned!
[Paper](https://arxiv.org/pdf/2403.03864.pdf) | [Website](https://algopuzzlevqa.github.io/)

## PuzzleVQA

Large multimodal models extend the impressive capabilities of large language models by integrating multimodal understanding abilities. However, it is not clear how they can emulate the general intelligence and reasoning ability of humans. As recognizing patterns and abstracting concepts are key to general intelligence, we introduce PuzzleVQA, a collection of puzzles based on abstract patterns. With this dataset, we evaluate large multimodal models with abstract patterns based on fundamental concepts, including colors, numbers, sizes, and shapes. Through our experiments on state-of-the-art large multimodal models, we find that they are not able to generalize well to simple abstract patterns. Notably, even GPT-4V cannot solve more than half of the puzzles. To diagnose the reasoning challenges in large multimodal models, we progressively guide the models with our ground truth reasoning explanations for visual perception, inductive reasoning, and deductive reasoning. Our systematic analysis finds that the main bottlenecks of GPT-4V are weaker visual perception and inductive reasoning abilities. Through this work, we hope to shed light on the limitations of large multimodal models and how they can better emulate human cognitive processes in the future.

### Example Puzzle

The figure below shows an example question which involves the color concept in PuzzleVQA, and an incorrect answer from
GPT-4V. There are generally three stages that can be observed in the solving process: visual perception (blue),
inductive reasoning (green), and deductive reasoning (red). Here, the visual perception was incomplete, causing a
mistake during deductive reasoning.

<img src="/img/example.png" alt="" width="600" height="1000">

### Puzzle Components

The figure below shows an illustration example of components (top) and reasoning explanations (bottom) for abstract
puzzles in PuzzleVQA. To construct each puzzle instance, we first define the layout and pattern of a multimodal
template, and populate the
template with suitable objects that demonstrate the underlying pattern. For interpretability, we also construct ground
truth reasoning explanations to interpret the puzzle and explain the general solution stages.

![](/img/components.png)

### Puzzle Taxonomy

The figure below shows the taxonomy of abstract puzzles in PuzzleVQA with sample questions, based on fundamental
concepts
such as colors and size. To enhance diversity, we design both single-concept and dual-concept puzzles.

![](/img/taxonomy.png)

### Evaluation Results

We report the main evaluation results on single-concept and dual-concept puzzles in Table 1 and Table 2 respectively.
The evaluation results for single-concept puzzles, as shown in Table 1 reveal notable differences in performance among
the open-source and closed-source models. GPT-4V stands out with the highest average score of 46.4, demonstrating
superior abstract pattern reasoning on single-concept puzzles such as numbers, colors, and size. It particularly excels
in the "Numbers" category with a score of 67.5, far surpassing other models, which may be due to its advantage in math
reasoning tasks (Yang et al., 2023). Claude 3 Opus follows with an overall average of 39.4, showing its strength in
the "Shapes" category with a top score of 44.5. The other models, including Gemini Pro and LLaVA-13B trail behind with
averages of 34.5 and 27.5 respectively, performing similarly to the random baseline on several categories.

In the evaluation on dual-concept puzzles, as shown in Table 2, GPT-4V stands out again with the highest average score
of 45.5. It performed particularly well in categories such as "Colors & Numbers" and "Colors & Size" with a score of
56.0 and 55.0 respectively. Claude 3 Opus closely follows with an average of 43.7, showing strong performance in "
Numbers & Size" with the highest score of 34.0. Interestingly, LLaVA-13B, despite its lower overall average of 31.1,
scores the highest in the "Size & Shapes" category at 39.0. Gemini Pro, on the other hand, has a more balanced
performance across categories but with a slightly lower overall average of 30.1. Overall, we find that models perform
similarly on average for single-concept and dual-concept patterns, which suggests that they are able to relate multiple
concepts such as colors and numbers together.

![](/img/results.png)


### Citation

```
@misc{chia2024puzzlevqa,
      title={PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns}, 
      author={Yew Ken Chia and Vernon Toh Yan Han and Deepanway Ghosal and Lidong Bing and Soujanya Poria},
      year={2024},
      eprint={2403.13315},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Algorithmic Puzzles

We introduce the novel task of multimodal puzzle solving, framed within the context of visual question-answering. We present a new dataset, AlgoPuzzleVQA designed to challenge and evaluate the capabilities of multimodal language models in solving algorithmic puzzles that necessitate both visual understanding, language understanding, and complex algorithmic reasoning. We create the puzzles to encompass a diverse array of mathematical and algorithmic topics such as boolean logic, combinatorics, graph theory, optimization, search, etc., aiming to evaluate the gap between visual data interpretation and algorithmic problem-solving skills. The dataset is generated automatically from code authored by humans. All our puzzles have exact solutions that can be found from the algorithm without tedious human calculations. It ensures that our dataset can be scaled up arbitrarily in terms of reasoning complexity and dataset size. Our investigation reveals that large language models (LLMs) such as GPT4V and Gemini exhibit limited performance in puzzle-solving tasks. We find that their performance is near random in a multi-choice question-answering setup for a significant number of puzzles. The findings emphasize the challenges of integrating visual, language, and algorithmic knowledge for solving complex reasoning problems.


### Visual Features of the Puzzles

The configuration of the puzzle/problem is shown as an image, which constitutes its visual context. We identify the following fundamental aspects of the visual context that influence the nature of the puzzles:

 - Colour
 - Position
 - Shape/Size
 - Text

<p align="center">
  <img src=img/teaser_visual.png />
</p>


### Algorithmic Features of the Puzzles

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


### Dataset

The dataset is available [here](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA/data) in [these format](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA#dataset). We created a total of 18 different puzzles spanning various algorithmic and mathematical topics. Many of these puzzles are popular in various recreational or academic settings.

In total, we have 1800 instances from the 18 different puzzles. These instances are analogous to different *test cases* of the puzzle, i.e.  they have different input combinations, initial and goal states, etc. Reliably solving all the instances would require finding the exact algorithm to use and then applying it accurately. This is akin to how we verify the accuracy of a computer program aiming to solve a particular task through a broad range of test cases.

We currently consider the full dataset as an **evaluation-only** benchmark. The detailed examples of all puzzles are shown [here](https://github.com/declare-lab/puzzle-reasoning/blob/master/puzzles.md).

Instructions for generating the dataset can be found [here](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA#dataset-generation). The number of instances and the difficulty of the puzzles can be scaled arbitrarily to any desired size or level.


### Ontology

The ontological categorization of the puzzles are as follows:

<p align="center">
  <img src=img/ontology.png />
</p>


### Experiments

The experimental setup and scripts can be found in the [AlgoPuzzleVQA](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA) directory.


### Citation

Please consider citing the following article if you found our work useful:

```bibtex
@article{ghosal2024algopuzzlevqa,
  title={Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning},
  author={Ghosal, Deepanway and Han, Vernon Toh Yan and Chia, Yew Ken and and Poria, Soujanya},
  journal={arXiv preprint arXiv:2403.03864},
  year={2024}
}
```
