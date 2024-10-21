
import json
import csv


# Column headers
columns = ["Method", "1", "10", "25", "50", "60", "70", "80", "90", "100"]

# Function to write the formatted CSV
def write_to_csv(data, filename="output.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the headers
        writer.writerow([""] + columns)  # First row, with blank first column for the first section
        
        # Iterate over the sections
        for section, metrics in data.items():
            writer.writerow([f"{section}: All results are averaged with 3 runs"])  # Add section header
            for metric_name, methods in metrics.items():
                writer.writerow([metric_name])  # Sub-header
                for method, results in methods.items():
                    # Write each method's results row
                    row = [method] + results
                    writer.writerow(row)
            writer.writerow([])  # Add a blank line between sections
with open('analysis.json','r') as f:
    data=json.load(f)
# Create the CSV
write_to_csv(data)

# Process the data and write to CSV
