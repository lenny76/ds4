# DwarfStar LAN Web UI

This dependency-free web UI exposes DwarfStar chat and its OpenAI-compatible
API to trusted devices on a local network. The model server remains bound to
localhost; this process is the only LAN-facing service.

## Start

### One command

Build the CPU server, point `DS4_MODEL` at the GGUF, and run:

```sh
make cpu
DS4_MODEL=/path/to/model.gguf webui/run.sh
```

The launcher starts the model server on localhost, waits for model loading and
weight warmup to finish, then exposes the UI on port 8080. `Ctrl+C` stops both
processes. It uses 40 CPU workers and a 4096-token context by default. Override
these with `DS4_THREADS` and `DS4_CTX`; set `DS4_WARM_WEIGHTS=0` to skip the
startup warmup. `DS4_UI_API_KEY` protects the LAN-facing API when set.

### Separate processes

Start DwarfStar on the Linux model host:

```sh
DS4_CPU_V41_EXPERIMENTAL=1 ./ds4-server --cpu -t 48 \
  -m /path/to/DeepSeek-V4.1-Flash-Q2.gguf \
  --ctx 4096 --host 127.0.0.1 --port 8000
```

In a second terminal, choose a key and start the UI:

```sh
export DS4_UI_API_KEY='replace-with-a-long-random-key'
python3 webui/server.py --host 0.0.0.0 --port 8080
```

Open `http://SERVER_LAN_IP:8080`, select **Impostazioni**, and enter the same
key. Conversations and settings are stored only in that browser's local
storage.

API clients can use the same LAN address and key:

```sh
curl http://SERVER_LAN_IP:8080/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-key'
```

The proxy supports all current `/v1/*` routes and streams responses without
buffering. Set `--upstream` if `ds4-server` is not at
`http://127.0.0.1:8000`. The UI can run without a key, but every device that
can reach port 8080 can then use the model.

Restrict port 8080 to the LAN in the host firewall. For example with UFW and a
`192.168.1.0/24` network:

```sh
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp
```

Replace the subnet with the server's actual LAN subnet. Do not expose this
service through router port forwarding.

## Run the tests

```sh
python3 -m unittest discover -s webui/tests -v
```
