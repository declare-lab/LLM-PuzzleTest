# Multimodal Puzzle Reasoning

## Algorithmic Puzzles

We introduce the novel task of multimodal puzzle solving, framed within the context of visual question-answering. We present a new dataset, AlgoPuzzleVQA designed to challenge and evaluate the capabilities of multimodal language models in solving algorithmic puzzles that necessitate both visual understanding, language understanding, and complex algorithmic reasoning. We create the puzzles to encompass a diverse array of mathematical and algorithmic topics such as boolean logic, combinatorics, graph theory, optimization, search, etc., aiming to evaluate the gap between visual data interpretation and algorithmic problem-solving skills. The dataset is generated automatically from code authored by humans. All our puzzles have exact solutions that can be found from the algorithm without tedious human calculations. It ensures that our dataset can be scaled up arbitrarily in terms of reasoning complexity and dataset size. Our investigation reveals that large language models (LLMs) such as GPT4V and Gemini exhibit limited performance in puzzle-solving tasks. We find that their performance is near random in a multi-choice question-answering setup for a significant number of puzzles. The findings emphasize the challenges of integrating visual, language, and algorithmic knowledge for solving complex reasoning problems.

## Dataset

Our dataset and the code for generating the dataset is available in the [AlgoPuzzleVQA](https://github.com/declare-lab/puzzle-reasoning/tree/master/AlgoPuzzleVQA) directory. We created a total of 18 different puzzles spanning various algorithmic and mathematical topics. Many of these puzzles are popular in various recreational or academic settings. Examples of puzzles in our dataset include [Tower of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi), [Four Colour Theorem](https://en.wikipedia.org/wiki/Four_color_theorem), [Mutilated chessboard](https://en.wikipedia.org/wiki/Mutilated_chessboard_problem), etc.

Detailed examples of all puzzles are shown [here](https://github.com/declare-lab/puzzle-reasoning/blob/master/puzzles.md).

## Visual Features of the Puzzles

The configuration of the puzzle/problem is shown as an image, which constitutes its visual context. We identify the following fundamental aspects of the visual context that influence the nature of the puzzles:

 - Colour
 - Position
 - Shape/Size
 - Text

<p align="center">
  <img src=img/teaser_visual.pdf />
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
  <img src=img/teaser_algorithm.pdf />
</p>

The algorithmic categories are not mutually exclusive, as we need to use two or more categories to derive the answer for most puzzles.


## Ontology

The ontological categorization of the puzzles are as follows:

<p align="center">
  <img src=img/ontology.png />
</p>