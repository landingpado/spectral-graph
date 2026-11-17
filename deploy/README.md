# Deploying the explorer on a small VPS behind Cloudflare

One droplet runs `python -m lawgraph.serve` (both pages and the API); a Cloudflare Tunnel gives it a public
HTTPS hostname. The build stays on the workstation: the droplet only serves what you upload.

Replace `DROPLET_IP` and `explorer.yourdomain.com` throughout. Droplet: Ubuntu 24.04, 2 vCPU / 4 GB / 80 GB,
created with your SSH key.

Checked locally: `environment.yml` builds an env that runs the whole explorer, and the server copes with
2 threads / 1 GB of database memory. Not checked: anything on a real droplet or in Cloudflare.

## 1. Prepare the droplet (once, as root)

    ssh root@DROPLET_IP
    apt update && apt -y upgrade
    adduser --disabled-password --gecos "" explorer
    rsync --archive --chown=explorer:explorer ~/.ssh /home/explorer     # same SSH key for the new user
    ufw allow OpenSSH && ufw --force enable                             # nothing but SSH is reachable
    exit

`ssh explorer@DROPLET_IP` should now log you in.

## 2. Upload (from the workstation)

    cd "/home/sean/PycharmProjects/Spectral_Spirit/process/spectral graph"
    rsync -avP --exclude '__pycache__' lawgraph viewer deploy explorer@DROPLET_IP:app/
    rsync -avP --append-verify graph.duckdb explorer@DROPLET_IP:app/

The first command sends 64 files, 90 MB. The second sends the 59 GB database: about 80 minutes at 100 Mbit/s
upload, 6.5 hours at 20 Mbit/s. If it is interrupted, run the same command again and it resumes.
Step 3 can be done while it runs.

## 3. Python environment (on the droplet, as explorer)

    ssh explorer@DROPLET_IP
    curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
    ~/miniforge3/bin/conda env create -f ~/app/deploy/environment.yml

This creates `auto-spectral` with the same Python and package versions the server uses on the workstation.

## 4. Try it privately

On the droplet:

    cd ~/app
    ~/miniforge3/envs/auto-spectral/bin/python -m lawgraph.serve --db graph.duckdb --port 8080 --no-open

On the workstation, in another terminal:

    ssh -N -L 8081:localhost:8080 explorer@DROPLET_IP

Open http://localhost:8081 and http://localhost:8081/years.html. Nothing is public yet: this goes through
SSH. In a second session on the droplet, `bash ~/app/deploy/check.sh` fetches both pages, the data and every
API endpoint and prints ok / FAIL for each. Ctrl-C the server and the tunnel when you have seen it work.

## 5. Run it as a service (as root)

    ssh root@DROPLET_IP
    cp /home/explorer/app/deploy/explorer.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now explorer
    systemctl status explorer --no-pager
    bash /home/explorer/app/deploy/check.sh

It now starts on boot and restarts if it fails. Logs: `journalctl -u explorer -f`.

## 6. Cloudflare Tunnel (public HTTPS, no open ports, no certificates)

1. Cloudflare dashboard, Zero Trust, Networks, Tunnels: create a tunnel of type Cloudflared and name it.
2. Choose Debian, 64-bit. Copy the install command it shows and paste it into the root shell on the droplet.
3. When the connector shows as connected, add a public hostname: subdomain `explorer`, your domain,
   service type HTTP, URL `localhost:8080`.
4. Open https://explorer.yourdomain.com, and from the workstation run
   `bash deploy/check.sh https://explorer.yourdomain.com`.

The server listens on 127.0.0.1 only, which is what the tunnel connects to.

## 7. Rate-limit the API

In the domain's Security settings (WAF, rate limiting rules): when the hostname is
`explorer.yourdomain.com` and the path starts with `/api/`, block an IP that makes more than about
50 requests in 10 seconds. The app itself does not limit requests.

## 8. Updating later (from the workstation)

    # after a rebuild of viewer/data
    rsync -avP --delete viewer/data/ explorer@DROPLET_IP:app/viewer/data/
    # after a code change
    rsync -avP --exclude '__pycache__' lawgraph viewer deploy explorer@DROPLET_IP:app/
    ssh root@DROPLET_IP systemctl restart explorer

## Known rough edges

- Static files are served with `Cache-Control: no-cache`, so Cloudflare does not hold them at the edge and
  every visitor's 86 MB comes from the droplet.
- `/api/docs` is public.
- The API reads only about 3 GB of the database; a slim copy would turn the upload into minutes.

## Showing it inside another site

The explorer can be framed in a page of a different site (same Cloudflare account or not). The page keeps
its own header and gives the rest of the window to the frame:

    <body class="explorer-page">
      <!-- the site's header -->
      <main class="stage">
        <iframe src="https://explorer.yourdomain.com/" title="Citation-space explorer"></iframe>
      </main>
    </body>

    .explorer-page { display: flex; flex-direction: column; height: 100dvh; }
    .stage { flex: 1; min-height: 0; background: #000; }
    .stage iframe { display: block; width: 100%; height: 100%; border: 0; }

`preview_in_glasscanon.png` is this layout in a local build of the GlassCanon site (1400 x 900: a 95 px
header, the frame fills the remaining height, the page itself does not scroll).
