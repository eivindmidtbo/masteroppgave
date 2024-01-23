# Utils - alphabetical number functions

# Helper functions for giving names to the trajectories. Alphabetical incremental counter


def __increment_chr(c: str) -> str:
    return chr(ord(c) + 1) if c != "Z" else "A"


def increment_alphabetical(s: str) -> str:
    lpart = s.rstrip("Z")
    number_replacements = len(s) - len(lpart)
    new_s = lpart[:-1] + __increment_chr(lpart[-1]) if lpart else "A"
    new_s += "A" * number_replacements
    return new_s


# Functions to get alphabetical numbers from normal numbers


def _get_char_value(key: int, base: int) -> str:
    if key >= 676:
        raise ValueError("Error: Too big number to be converted to alphabetic number")

    return chr(ord(base) + key // 26) + chr(ord(base) + key % 26)


def get_alphabetical_value(hash: list[int] | int) -> str:
    """
    Returns a string representing a hash output
    Big letters represent the lat value and small letters represent the long value
    """

    out_value = ""

    if isinstance(hash, int):
        out_value += _get_char_value(hash, "A")

    else:
        for i, key in enumerate(hash):
            match i:
                case 0:
                    out_value += _get_char_value(key, "A")
                case 1:
                    out_value += _get_char_value(key, "a")
                case _:
                    raise ("Exception during alphabetical value conversion")

    return out_value


if __name__ == "__main__":
    assert get_alphabetical_value([1, 13]) == "ABan"
    assert get_alphabetical_value([5, 32]) == "AFbg"
    assert get_alphabetical_value([32, 2]) == "BGac"
    assert get_alphabetical_value([35, 66]) == "BJco"
    assert get_alphabetical_value([675, 675]) == "ZZzz"
    assert get_alphabetical_value(0) == "AA"
    assert get_alphabetical_value(1) == "AB"
    assert get_alphabetical_value(35) == "BJ"
    assert get_alphabetical_value(675) == "ZZ"

    print("All tests returned true")
