import re
import sys
import pyperclip

def get_indent(line):
    """Returns the indentation of a given line."""
    return len(line) - len(line.lstrip())

def adjust_indentation(method_body, base_indent):
    """Adjusts the indentation of the method body to match the base indentation level."""
    lines = method_body.splitlines()
    if not lines:
        return ""

    # Find the minimum indentation in the new method body
    min_indent = min(get_indent(line) for line in lines if line.strip())

    adjusted_lines = []
    for line in lines:
        if line.strip():
            # Remove the minimum indentation and add the base indentation
            adjusted_line = " " * base_indent + line[min_indent:]
            adjusted_lines.append(adjusted_line)
        else:
            adjusted_lines.append("")

    return "\n".join(adjusted_lines)

def replace_method_in_file(filename, method_name, new_method_body):
    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Regular expression to find the method definition and its body
        method_regex = rf"(^\s*def\s+{method_name}\s*\([^\)]*\)\s*:\s*)(([\s\S]*?)(?=\n\S|$))"
        match = re.search(method_regex, content, re.MULTILINE)
        
        if not match:
            print(f"Error: Method '{method_name}' not found in '{filename}'.")
            return

        # Get the base indentation level (indentation of the 'def' line)
        base_indent = get_indent(match.group(1))

        # Adjust the indentation of the new method body
        adjusted_method_body = adjust_indentation(new_method_body, base_indent + 4)  # Add 4 spaces for inner indentation

        # Rebuild the method definition with the original signature and new body
        new_method = f"{match.group(1)}\n{adjusted_method_body}"

        # Replace the old method with the new one
        new_content = content[:match.start()] + new_method + content[match.end():]

        with open(filename, 'w') as file:
            file.write(new_content)

        print(f"Method '{method_name}' successfully replaced in '{filename}'.")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python replace_method.py <filename> <method_name>")
        sys.exit(1)

    filename = sys.argv[1]
    method_name = sys.argv[2]

    # Get the content from the clipboard and remove any `\r` (carriage return)
    new_method_body = pyperclip.paste().replace('\r', '')
    if not new_method_body:
        print("Error: Clipboard is empty.")
        sys.exit(1)

    # Replace the method in the file
    replace_method_in_file(filename, method_name, new_method_body)

if __name__ == "__main__":
    main()
