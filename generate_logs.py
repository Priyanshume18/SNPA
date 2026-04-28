import csv

INPUT_FILE = "access_logs.csv"
OUTPUT_FILE = "access_logs_filtered.csv"  # change to same file if you want overwrite
THRESHOLD = 0.7


def filter_logs(input_file, output_file, threshold):
    filtered_rows = []
    last_plate = None

    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 5:
                continue  # skip bad rows

            timestamp, plate, status, confidence, frame = row

            try:
                confidence = float(confidence)
            except ValueError:
                continue

            # ✅ Filter by confidence
            if confidence < threshold:
                continue

            # ✅ Remove consecutive duplicates
            if plate == last_plate:
                continue

            filtered_rows.append(row)
            last_plate = plate

    # write output
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(filtered_rows)


if __name__ == "__main__":
    filter_logs(INPUT_FILE, OUTPUT_FILE, THRESHOLD)
    print("Filtering complete.")