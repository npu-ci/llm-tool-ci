import re
import json
import argparse

PATTERN = r"<!--{}-{}-->"


def update_table(file_path, table_path, table_item, tool_name):
    with open(file_path, "r") as f:
        text = f.read()

    if table_path:
        with open(table_path, "r") as f:
            table = json.load(f)

        for model, support in table.items():
            pattern = PATTERN.format(tool_name, model)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                past_state = text[match.start() : match.end() + 1]
                current_state = past_state[:-1] + ("✅" if support else " ")
                text = text.replace(past_state, current_state)

        with open(file_path, "w") as f:
            f.write(text)
    elif table_item:
        model, support = table_item.split(":")
        pattern = PATTERN.format(tool_name, model)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            past_state = text[match.start() : match.end() + 1]
            current_state = past_state[:-1] + ("✅" if int(support)==0 else " ")
            text = text.replace(past_state, current_state)
        with open(file_path, "w") as f:
            f.write(text)



parser = argparse.ArgumentParser()
parser.add_argument(
    "--file-path", "-f", type=str, default=None, help="Specify a file path."
)
parser.add_argument(
    "--table-path", "-t", type=str, default=None, help="Specify a table file path."
)
parser.add_argument(
    "--tool-name",
    "-n",
    type=str,
    default=None,
    choices=["transformers", "accelerate", "peft", "fschat"],
    help="Specify a tool name.",
)
parser.add_argument(
    "--table-item", "-ti", type=str, default=None, help="Specify a table item key:value str."
)


if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file_path
    table_path = args.table_path
    tool_name = args.tool_name
    table_item = args.table_item

    if not table_item and not table_path:
        raise ValueError("Must configure one of table_item or table_path.")
    
    if table_item and (type(table_item) is not str):
        raise ValueError("table_item must be a key:value str.")
    
    if file_path:
        if type(file_path) is not str:
            raise ValueError("file-path or table_path must be a file path of str type.")
        if not file_path.endswith(".md"):
            raise ValueError("file-path must be a Markdown file.")
    
    if table_path:
        if type(table_path) is not str:
            raise ValueError("file-path or table_path must be a file path of str type.")
        if not table_path.endswith(".json"):
          raise ValueError("file-path must be a Json file.")


    update_table(file_path, table_path, table_item, tool_name)
