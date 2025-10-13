import os
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

def read_text_file(file_path):
    """
    Read a text file line by line using different methods
    """
    print("Method 1: Using readlines()")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line.strip())

    print("\nMethod 2: Using read() and split()")
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        for line in lines:
            print(line)

    print("\nMethod 3: Direct line iteration")
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())

def read_json_file(file_path):
    """
    Read a JSON file using different methods
    """
    print("Method 1: Using json.load()")
    with open(file_path, 'r') as file:
        data = json.load(file)
        print(json.dumps(data, indent=2))

    print("\nMethod 2: Using pandas")
    # Convert the JSON to a format pandas can handle better
    df = pd.DataFrame.from_dict(data, orient='index', columns=['quantity'])
    df.index.name = 'item'
    print(df)

def read_csv_file(file_path):
    """
    Read a CSV file using different methods
    """
    print("Method 1: Using csv module")
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            print(row)

    print("\nMethod 2: Using pandas")
    df = pd.read_csv(file_path)
    print(df)

def read_xml_file(file_path):
    """
    Read an XML file using different methods
    """
    print("Method 1: Using ElementTree")
    tree = ET.parse(file_path)
    root = tree.getroot()
    for item in root.findall("grocery_item"):
        name = item.find("name").text
        price = item.find("price").text
        print(f"Item: {name}, Price: ${price}")

    print("\nMethod 2: Using pandas")
    df = pd.read_xml(file_path)
    print(df)

# Example usage
if __name__ == "__main__":
    # Make sure these files exist in your workspace
    text_file = "groceries.txt"
    json_file = "groceries.json"
    csv_file = "groceries.csv"
    xml_file = "groceries.xml"

    print("=== Reading Text File ===")
    if os.path.exists(text_file):
        read_text_file(text_file)

    print("\n=== Reading JSON File ===")
    if os.path.exists(json_file):
        read_json_file(json_file)

    print("\n=== Reading CSV File ===")
    if os.path.exists(csv_file):
        read_csv_file(csv_file)

    print("\n=== Reading XML File ===")
    if os.path.exists(xml_file):
        read_xml_file(xml_file)