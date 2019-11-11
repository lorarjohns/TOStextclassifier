import json 
import re
from pathlib import Path

path = Path("tosdr_api.json").resolve()
with open(path) as f:
    json = json.load(f)

regex = re.compile(r"tosdr\/review\/[\w\-\.]+")
for key in json.keys():
    if re.match(regex, key):
        try:
            if len(json[key]["documents"]) > 0:
                print(json[key]["documents"])
        except:
            pass
