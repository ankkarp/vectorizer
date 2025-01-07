from cairosvg import svg2png

with open(f"test.svg", "r") as f:
    svg = f.read()
    svg2png(bytestring=(svg), write_to=f"test.png")