"""
Similarity mappings module.

The module provides functionality to store and manage a similarity mappings between
records of two datasets.
"""

from rapidfuzz import distance, fuzz
import numpy
import typing


def notmissing(x, y) -> float:
    """0 if either one of None, np.nan, pd.nan or '' appears."""
    return 1.0 if x and y and not (isinstance(x, (float, int)) and numpy.isnan(x)) and not (isinstance(y, (float, int)) and numpy.isnan(y)) else 0.0


def notzero(x, y) -> float:
    """0 if either one of None, np.nan, pd.nan or '' appears."""
    return 1.0 if x and y and not any(i==0 for i in (x,y))else 0.0


def discrete(
    x: typing.Union[int, str, typing.AnyStr], y: typing.Union[int, str, typing.AnyStr]
) -> float:
    """Discrete similarity function."""
    return 1.0 if x == y else 0.0


def euclidean(x: typing.Union[float, int], y: typing.Union[float, int]) -> float:
    """Euclidean similarity function."""
    return 1.0 / (1.0 + abs(x - y))


def gaussian(x: typing.Union[float, int], y: typing.Union[float, int]) -> float:
    """Gaussian similarity function."""
    return numpy.exp(-((x - y) ** 2) / 2.0)


def available_similarities() -> typing.Dict[str, typing.Callable]:
    """Return the list of available similarities."""
    return {
        "basic_ratio": fuzz.ratio,
        "damerau_levenshtein": distance.DamerauLevenshtein.normalized_similarity,
        "discrete": discrete,
        "euclidean": euclidean,
        "gaussian": gaussian,
        "hamming": distance.Hamming.normalized_similarity,
        "indel": distance.Indel.normalized_similarity,
        "jaro": distance.Jaro.normalized_similarity,
        "jaro_winkler": distance.JaroWinkler.normalized_similarity,
        "lcsseq": distance.LCSseq.normalized_similarity,
        "levenshtein": distance.Levenshtein.normalized_similarity,
        "notmissing": notmissing,
        "notzero": notzero,
        "osa": distance.OSA.normalized_similarity,
        "partial_ratio": fuzz.partial_ratio,
        "partial_ratio_alignment": fuzz.partial_ratio_alignment,
        "partial_token_ratio": fuzz.partial_token_ratio,
        "partial_token_set_ratio": fuzz.partial_token_set_ratio,
        "partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
        "postfix": distance.Postfix.normalized_similarity,
        "prefix": distance.Prefix.normalized_similarity,
        "token_ratio": fuzz.token_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
    }


class SimilarityMap:
    """
    Similarity map class.

    The class stores a collection of associations between the records of two datasets.

    Attributes:
        instructions (dict): The similarity map instructions.
        lcols (list[str]): The left columns.
        rcols (list[str]): The right columns.
        sims (list[str]): The similarity functions.
    """

    def __init__(self, instructions):
        """Initialize a similarity map object.

        Args:
            instructions: The similarity map instructions.
        """
        if not isinstance(instructions, dict):
            raise ValueError("Input instructions must be a dictionary.")
        for key, value in instructions.items():
            if not isinstance(key, str):
                raise ValueError("Association key must be a string.")
            if not isinstance(value, list):
                raise ValueError(
                    "Association values must be a list. "
                    f"Instead got type {type(value)}."
                )
        self.instructions = instructions
        self.lcols = []
        self.rcols = []
        self.sims = []
        for association, similarities in self.instructions.items():
            parts = association.split("~")
            assert len(parts) == 1 or len(parts) == 2
            lcol = parts[0].strip()
            rcol = parts[1].strip() if len(parts) == 2 else lcol
            for similarity in similarities:
                self.lcols.append(lcol)
                self.rcols.append(rcol)
                self.sims.append(similarity)

    def no_associations(self) -> int:
        """Return the number of associations of the map."""
        return len(self.instructions)

    def association_sizes(self) -> typing.List[int]:
        """Return then number of similarities used by each association."""
        return [len(instruction) for instruction in self.instructions.values()]

    def association_offsets(self) -> typing.List[int]:
        """Return association offsets.

        Return the starting column offset of each association in the
        similarity matrix
        """
        return numpy.cumsum([0] + self.association_sizes()).tolist()[0:-1]

    def association_names(self) -> typing.List[str]:
        """Return a unique name for each association in the similarity map."""
        names = []
        for key in self.instructions.keys():
            parts = [p.strip() for p in key.split("~")]
            names.append(parts[0] if len(parts) == 1 else f"{parts[0]}_{parts[1]}")
        return names

    def keys(self) -> typing.List[str]:
        """Return a unique key for each similarity map entry.

        Combine association with similarity names and return them.
        """
        keys = []
        for key, value in self.instructions.items():
            parts = [p.strip() for p in key.split("~")]
            name = parts[0] if len(parts) == 1 else f"{parts[0]}_{parts[1]}"
            for similarity in value:
                keys.append(f"{name}_{similarity}")
        return keys

    def __iter__(self) -> typing.Iterator:
        """Iterate over the similarity map."""
        return zip(self.lcols, self.rcols, self.sims)

    def __len__(self) -> int:
        """Return the number of items in the similarity map."""
        return len(self.lcols)

    def __getitem__(self, index) -> typing.Tuple[str, str, str]:
        """Return the item at the given index."""
        return self.lcols[index], self.rcols[index], self.sims[index]

    def __str__(self) -> str:
        """Return a string representation of the similarity map."""
        items = "\n  ".join([str(item) for item in self])
        return f"{self.__class__.__name__}[\n  {items}]"
