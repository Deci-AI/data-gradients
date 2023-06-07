import seaborn

PALETTE_NAME = "pastel"
PALETTE = seaborn.color_palette(PALETTE_NAME, 100)
LABELS_PALETTE = {"train": PALETTE[0], "val": PALETTE[1], "test": PALETTE[2]}
