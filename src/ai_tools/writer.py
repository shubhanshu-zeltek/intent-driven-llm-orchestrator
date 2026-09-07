from langchain_core.tools import tool


@tool
def write_text(data: str | list[str], file_name: str) -> bool:
    """
    Write data to a text file. If the file already exists then overwrites it.
    Returns True if data is written successfully, otherwise returns False.

    Use this tool whenever a data needs to be saved in a text file.
    """
    try:
        if isinstance(data, list):
            data = '\n'.join(data)
        
        with open(file_name, 'w') as file:
            file.write(data)
        return True
    except Exception as e:
        print(f"Error writing data to {file_name}: {e}")
        return False



