import csv
import gzip
import io
import json
import pandas as pd
import paramiko
import variables as var


desc_path = f"descriptions/nyc_open_data/{var.desc_model}/0.json"


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

    # Read table descriptions
    with open(desc_path, "r") as f:
        for line in f:
            tab_json = json.loads(line)

    # Read the table (stored on the remote server)
    with sftp.open(tab_json["path"], "rb") as remote_file:
        compressed_data = remote_file.read()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed_data), mode="rb") as gz:
        text_stream = io.TextIOWrapper(gz, encoding="utf-8")
        reader = csv.reader(text_stream, delimiter="\t")
        tab_content = [row for row in reader]
        tab_header = tab_content[0]
        tab_content = tab_content[1:]
        tab_content_cols = list(map(set, zip(*tab_content)))

    print(pd.DataFrame(tab_content, columns=tab_header))
    print(tab_json["summaries"])

    # Close all connections
    sftp.close()
    target_client.close()
    jump_client.close()


if __name__ == "__main__":
    main()
