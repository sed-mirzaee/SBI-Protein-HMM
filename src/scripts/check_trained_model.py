import numpy as np

from src.models.amortized_posterior import (
    predict_from_sequence,
)


def main() -> None:
    sequence = "ARNDCEQGHILKMFPSTWYV" * 3

    probabilities = predict_from_sequence(
        sequence
    )

    print("Shape:", probabilities.shape)
    print("First rows:")
    print(probabilities[:5])

    assert probabilities.shape == (
        len(sequence),
        2,
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert np.allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    print("Trained-model inference works.")


if __name__ == "__main__":
    main()