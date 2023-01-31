import numpy as np


def store_ans(answers, file_name="answer.txt"):
    # This function ensures that the format of submission
    with open(file_name, "w") as f:
        for answer in answers:
            # Ensure that 6 decimals are used
            f.write("{:.6f}\n".format(answer))


if __name__ == "__main__":
    # assume it's the predicted accuracy for 100 datasets in total
    generated_answer = np.random.rand(100)
    # NOTE: Call the `store_ans` function for an iterable object,
    #       which contains your accuracy predictions. This function
    #       will write the predictions to a file.
    store_ans(answers=generated_answer)
