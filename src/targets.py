from src.config import ION_TYPES, CHARGES, MAX_FRAG_POS

def extract_targets(row):
    """
    Extract b/y ions (charge 1) for fragment positions 1..39.
    Returns:
        target: list of floats length 78
        mask: list of booleans length 78 (True = valid, False = ignore)
    """
    target = []
    mask = []

    for ion in ION_TYPES:
        for pos in range(1, MAX_FRAG_POS + 1):
            key = (ion, "1", str(pos))
            key_str = str(key)  # keys in dataset look like "('b', '1', '2')"

            val = row.get(key_str, -1.0)

            target.append(val)

            if val == -1.0:
                mask.append(False)  # ignore
            else:
                mask.append(True)   # valid

    return target, mask
