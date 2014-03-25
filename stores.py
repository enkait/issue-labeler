import json

class AppendStore:
    def __init__(self, file_path):
        self.f = open(file_path, "a+")

    def store(self, element):
        self.f.write(json.dumps(element)+"\n")

class OverwriteStore:
    def __init__(self, file_path):
        self.file_path = file_path

    def store(self, element):
        with open(self.file_path, "w") as f:
            f.write(json.dumps(element))

    def load(self):
        with open(self.file_path, "r") as f:
            return json.loads(f.read())

class MemoryStore:
    def __init__(self):
        self.L = []

    def store(self, element):
        self.L.append(element)

    def get_stored(self):
        return self.L

