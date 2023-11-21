import json
import argparse


def create_file(file_path):
    with open(file_path, "w") as f:
        json.dump({}, f)


def add_tag(file_path, add):
    with open(file_path, "r") as f:
        text = json.load(f)

    k, v = add.split(":")
    if v == "true":
        text.update({k: True})
    elif v == "false":
        text.update({k: False})
    else:
        raise ValueError("Value must be either 'true' or 'false'.")

    with open(file_path, "w") as f:
        json.dump(text, f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file-path", "-f", type=str, default=None, help="Specify a file path."
)
parser.add_argument(
    "--create", action="store_true", help="Add '--create' to create a new file."
)
parser.add_argument("--add", "-a", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file_path
    create = args.create
    add = args.add

    if type(file_path) is not str:
        raise ValueError("file-path must be a file path of str type.")

    if not file_path.endswith(".json"):
        raise ValueError("file-path must be a Json file.")

    if create:
        create_file(file_path)

    if add is not None:
        if ":" not in add:
            raise ValueError('The added tag must be the form of "model:true"')
        add_tag(file_path, add)
