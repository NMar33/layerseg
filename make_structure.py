import click
from typing import Tuple
from pathlib import Path



def parse_line(line: str) -> Tuple[int, str, str]:
    """
    Parses a line (str) that represents a file or directory path 
    in a specific format:

        data/
        - raw/
        -- img_1.jpg

    Returns a tuple containing:
        - the level of the current line (as determined by the number of hyphens)
        - the type of the file or directory ('file' or 'dir')
        - the name of the file or directory.

    Parameters:
        line (str): The line to be parsed.

    Returns:
        Tuple[int, str, str]: A tuple containing the level of the current line, the type of the 
        file or directory, and the name of the file or directory.

    Example usage:
        >>> parse_line('- data/')
        (1, 'dir', 'data')
        >>> parse_line('--- img_1.jpg')
        (3, 'file', 'img_1.jpg')
    """

    split_line = line.strip().rsplit(" ", maxsplit=1)
    if len(split_line) == 1:
        lvl = 0
        parsed_line = split_line[0]
    else:
        lvl = len(split_line[0])
        parsed_line = split_line[1]
    line_type = "dir" if parsed_line[-1] == "/" else "file"
    parsed_line = parsed_line.strip("/")

    return lvl, line_type, parsed_line


def parent_path_update(
        parent_path: Path, 
        parent_lvl: int, 
        line_lvl: int) -> Tuple[Path, int]:
    """
    Updates the parent path based on the level of the current line.

    Parameters:
        parent_path (Path): The current parent path.
        parent_lvl (int): The level of the current parent path.
        line_lvl (int): The level of the current line.

    Raises:
        ValueError: If the line level is more than one level higher than the parent level.

    Returns:
        - The updated parent path.
        - The updated parent lvl.
    """
    
    if line_lvl <= parent_lvl:
        short_parents = parent_lvl - line_lvl
        parent_lvl -= 1 + short_parents
        parent_path = parent_path.parents[short_parents]
    elif line_lvl > parent_lvl + 1:
        raise ValueError(f"The line level ({line_lvl}) cannot be more than one level higher than the parent level ({parent_lvl}).")
    return parent_path, parent_lvl


def make_structure(
        path_to_struct_file: str,
        path_make_struct) -> None:
    
    parent_lvl = -1
    parent_path = Path(path_make_struct)    
    with open(path_to_struct_file, "r") as inp_stream:
        for line in inp_stream:
            if line == "" or line.isspace():
                continue
            
            line_lvl, line_type, line_name = parse_line(line)
            parent_path, parent_lvl = parent_path_update(parent_path, parent_lvl, line_lvl)
            line_path = Path.joinpath(parent_path, line_name)
            if line_type == "dir":                
                line_path.mkdir(parents=False, exist_ok=True)
                parent_path = line_path
                parent_lvl += 1
            elif line_type == "file":
                open(line_path, "w").close()




@click.command(
    name="make_structure"
)
@click.option(
    "-sfp",
    "--struct_file_path",
    default="structure.md",
    type=str
)
@click.option(
    "-msp",
    "--make_struct_path",
    default="",
    type=str
)
def make_structure_command(
    struct_file_path: str,
    make_struct_path: str) -> None:
    make_structure(struct_file_path, make_struct_path)


if __name__ == "__main__":
    make_structure_command()