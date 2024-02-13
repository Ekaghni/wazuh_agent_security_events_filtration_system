import csv
import random
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data
def select_rows(data, num_rows):
    return random.choices(data, k=num_rows)
def write_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        fieldnames = ['count', 'command', 'pushing_to_server_required']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for idx, row in enumerate(data, start=1):
            row['count'] = str(idx)
            writer.writerow(row)
def main():
    original_csv = "agent_commands_t2.csv"
    new_rows = 50000

    original_data = read_csv(original_csv)
    selected_rows = select_rows(original_data, new_rows)
    write_csv(original_csv, selected_rows)

if __name__ == "__main__":
    main()
