import csv
import gzip
import io
import json
import math
import os
import paramiko
import random
import re
import time
import variables as var
from groq import Groq


def define_column_batches(cols, batch_size):
    n = len(cols)
    num_batches = math.ceil(n / batch_size)
    base_batch_size = n // num_batches
    remainder = n % num_batches

    indices = list(range(n))
    result = list()
    start = 0

    for i in range(num_batches):
        size = base_batch_size + (1 if i < remainder else 0)
        result.append(indices[start : start + size])
        start += size

    return result


def main():

    # Connect to the jump server
    jump_client = paramiko.SSHClient()
    jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump_client.connect(var.jump_host, port=var.jump_port, username=var.jump_user, password=var.jump_password)

    # Open a channel from jump server to target server
    jump_transport = jump_client.get_transport()
    channel = jump_transport.open_channel("direct-tcpip", (var.target_host, var.target_port), ("", 0))

    # Connect to the target server using the tunneled channel
    target_client = paramiko.SSHClient()
    target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target_client.connect(var.target_host, port=var.target_port, username=var.target_user, password=var.target_password, sock=channel)

    sftp = target_client.open_sftp()

    # Connect to the API of the LLM used to generate column descriptions
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Create the directory to store the column descriptions (if it does not exist yet)
    path_desc_dir = "descriptions/%s/%s/" % (var.corpus, var.desc_model)
    os.makedirs(path_desc_dir, exist_ok=True)

    tab_id = 0
    for i in range(len(var.path_corpus)):
        tab_context = ", ".join([x.replace("_", " ") for x in var.path_corpus[i].split("/")])
        path_corpus = var.path_data + var.path_corpus[i]
        tab_files = sorted(sftp.listdir(path_corpus))
        for j in range(len(tab_files)):
            # if tab_id % var.milestone == 0 and tab_id > 0:
            #     print("Descriptions created for %s tables." % tab_id)

            # Check whether the column descriptions for the current table already exist
            path_desc_file = path_desc_dir + "%s.json" % tab_id
            if os.path.isfile(path_desc_file):
                tab_id += 1
                continue

            # Read the table (stored on the remote server)
            with sftp.open(path_corpus + "/" + tab_files[j], "rb") as remote_file:
                compressed_data = remote_file.read()
            with gzip.GzipFile(fileobj=io.BytesIO(compressed_data), mode="rb") as gz:
                text_stream = io.TextIOWrapper(gz, encoding="utf-8")
                reader = csv.reader(text_stream, delimiter="\t")
                tab_content = [row for row in reader]
                tab_header = tab_content[0]
                tab_content = tab_content[1:]
                tab_content_cols = list(map(set, zip(*tab_content)))

            print(f"Generating descriptions for table {tab_id} (with {len(tab_header)} columns).")
            start_time = time.time()

            # Generate the descriptions for each batch
            batches = define_column_batches(tab_header, var.desc_batch_cols)
            descriptions = {k: list() for k in range((len(tab_header)))}  # {col_id: [descriptions]}
            summaries = {k: None for k in range((len(tab_header)))}  # {col_id: summary}

            for batch in batches:

                if len(batch) < len(tab_header):
                    front_cols = [x for x in range(len(tab_header))][:min(var.desc_front_cols, len(tab_header))]
                    back_cols = [x for x in range(len(tab_header))][max(len(tab_header) - var.desc_back_cols, 0):]
                    col_indices = (set(batch + front_cols + back_cols))
                else:
                    col_indices = batch

                # Produce the description for each column in the batch (multiple attempts)
                for _ in range(var.desc_attempts):

                    prompt = f"I have the following table, whose context is: '{tab_context}'\n\n"

                    sample_rows = random.sample(tab_content, min(var.desc_sample_rows, len(tab_content_cols[0])))

                    prompt += " | ".join([tab_header[k] for k in range(len(tab_header)) if k in col_indices])
                    prompt += "\n"
                    prompt += " | ".join(["---" for k in range(len(tab_header)) if k in col_indices])
                    prompt += "\n"
                    for row in sample_rows:
                        prompt += " | ".join([row[k] for k in range(len(tab_header)) if k in col_indices])
                        prompt += "\n"
                    prompt += "\n"

                    set_batch = set(batch)
                    col_samples = [random.sample(list(tab_content_cols[k]), min(len(tab_content_cols[k]), var.desc_col_samples))
                                   if k in set_batch else None for k in range(len(tab_header))]

                    prompt += "Provide a complete and essential description for the following columns, " + \
                              "in the format: 'Column {{index}} contains'.\n"

                    for col_id in batch:
                        prompt += f"- 'Column {col_id}', with header '{tab_header[col_id]}' " + \
                                  f"and values like: {str(col_samples[col_id])[1:-1]}.\n"

                    try:
                        message = client.chat.completions.create(model=var.desc_model, messages=[{"role": "user", "content": prompt}],
                                                                    temperature=0)
                        response = message.choices[0].message.content.lower()
                    except Exception as e:
                        time.sleep(5)

                    for col_id in batch:
                        match = re.search(rf"column {col_id} (.*?)\.", response)
                        descriptions[col_id].append(match.group(1) if match else None)

                # Produce the summary of the descriptions for each column

                prompt = ""
                batch_ctr = 0

                # Build the prompt
                for col_id in batch:

                    if len(descriptions[col_id]) == 0:
                        summaries[col_id] = None
                        continue

                    if batch_ctr > 0:
                        prompt += "--------------------------------------------------\n\n"

                    prompt += f"QUESTION {batch_ctr}\n\n"

                    prompt += "Produce a complete and essential summary of the following descriptions, " + \
                              f"in the format: 'Column {col_id} contains'.\n\n"
                    
                    for desc in descriptions[col_id]:
                        prompt += f"- Column {col_id} {desc}\n\n"

                    batch_ctr += 1

                try:
                    message = client.chat.completions.create(model=var.desc_model, messages=[{"role": "user", "content": prompt}],
                                                                temperature=0)
                    response = message.choices[0].message.content.lower()
                except Exception as e:
                    time.sleep(5)

                for col_id in batch:
                    match = re.search(rf"column {col_id} (.*?)\.", response)
                    if match:
                        summaries[col_id] = f"Column '{tab_header[col_id]}' " + match.group(1)

            # Store the descriptions and summaries for the current table in a JSONLines file
            with open(path_desc_file, "w") as f:
                tab_json = {"path": path_corpus + "/" + tab_files[j], "summaries": summaries, "descriptions": descriptions}
                json.dump(tab_json, f)

            end_time = time.time()
            print(f"Generation completed in {end_time - start_time} seconds.\n")

            exit(0)

            tab_id += 1

    # Close all connections
    sftp.close()
    target_client.close()
    jump_client.close()


if __name__ == "__main__":
    main()
