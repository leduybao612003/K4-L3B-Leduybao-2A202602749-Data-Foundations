import re, glob

for path in sorted(glob.glob("data/ecommerce-policy/*.md")):
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---"):
        continue
    parts = text.split("---", 2)
    fm = re.sub(r'^(\w+):\s*"(.*)"\s*$', r'\1: \2', parts[1], flags=re.M)
    new_text = "---" + fm + "---" + parts[2]
    if new_text != text:
        open(path, "w", encoding="utf-8", newline="\n").write(new_text)
        print("cleaned:", path)