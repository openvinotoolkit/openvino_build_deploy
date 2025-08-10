# How to select hallucination computation algorithm

Currently, two methods are availble: [deepeval](#use-deepeval-to-compute-hallucination-score) and [selfcheckgpt](#use-selfcheckgpt-to-compute-hallucination-score).

If you have an evaluation dataset (i.e. both question and correct answer), you can choose [deepeval](#use-deepeval-to-compute-hallucination-score). However, if you do not have a labeled dataset, you can choose [selfcheckgpt](#use-selfcheckgpt-to-compute-hallucination-score). It will compute hallucination score based on the output consistency.

# Use deepeval to compute hallucination score
## Prerequisite libraries
1. [deepeval](https://github.com/confident-ai/deepeval)
2. [Ollama](https://github.com/ollama/ollama/blob/main/README.md)

## How to set up
1. Install deepeval:
    ```
    pip install -U deepeval
    ```
2. Install Ollama:
    Please refer to [ollama](https://github.com/ollama/ollama/blob/main/README.md#ollama)

3. Run Ollama, taking `deepseek-r1` as an example:
    ```
    ollama run deepseek-r1
    ```
4. Set deepeval to use Ollama for evaluation:
    ```
    deepeval set-ollama deepseek-r1
    ```

## How to run the test
```
python test.py --personality /path/to/personality.yaml --check_type deepeval
```

## More to read
[deepeval hallucination](https://docs.confident-ai.com/docs/metrics-hallucination)

# Use selfcheckgpt to compute hallucination score
## Prerequisite libraries
1. [selfcheckgpt](https://github.com/potsawee/selfcheckgpt)

## How to set up and run the test
1. Install deepeval:
    ```
    pip install selfcheckgpt==0.1.7
    ```

2. Run test
    ```
    python test.py --personality /path/to/personality.yaml --check_type selfcheckgpt
    ```