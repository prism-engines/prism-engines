ECMWFdatastores.md (READY TO COPY INTO REPO)
# ECMWF Data Stores Client Configuration  
This guide explains how to configure authentication for the ECMWF / Copernicus Climate Data Store (CDS) using the new **ecmwf-datastores** client.

The client requires:

- The API root URL  
- Your personal API key (formatted as `UID:APIKEY`)

The client reads these values in the following order of precedence:

1. **Explicit arguments** provided when instantiating the client  
2. **Environment variables**  
3. **Configuration file** located at `~/.ecmwfdatastoresrc`  

---

## 1. Recommended Method: Create `~/.ecmwfdatastoresrc`

This is the standard configuration used by PRISM and most ECMWF tooling.

### **Create the configuration file**
```bash
nano ~/.ecmwfdatastoresrc

Paste the following (replace with your real UID and API key)
url: https://cds.climate.copernicus.eu/api
key: UID:APIKEY


Example format:

123456:abcd-ef12-3456-7890-fedcba987654

Save and exit

Press CTRL + O, then Enter

Press CTRL + X

Verify the file
cat ~/.ecmwfdatastoresrc


You should see the URL and key printed.

2. Alternative: Environment Variables

These can be set in your shell or permanently added to ~/.zshrc.

export ECMWF_DATASTORES_URL="https://cds.climate.copernicus.eu/api"
export ECMWF_DATASTORES_KEY="UID:APIKEY"


To make them persistent:

echo 'export ECMWF_DATASTORES_URL="https://cds.climate.copernicus.eu/api"' >> ~/.zshrc
echo 'export ECMWF_DATASTORES_KEY="UID:APIKEY"' >> ~/.zshrc
source ~/.zshrc

3. Alternative: Pass Credentials Programmatically

Useful when running isolated scripts or tests.

from ecmwfdatastores import Client

client = Client(
    url="https://cds.climate.copernicus.eu/api",
    key="UID:APIKEY"
)

4. Notes for PRISM Integration

PRISM fetchers expect credentials to be present in ~/.ecmwfdatastoresrc.

Do not commit real API keys to GitHub.

Your UID and API key come from the CDS "API key" section of your account.

5. Troubleshooting
Confirm that the config file is readable
ls -l ~/.ecmwfdatastoresrc

Test authentication manually
python - <<EOF
from ecmwfdatastores import Client
c = Client()
print("Client initialized:", c)
EOF


If the client initializes without error, your configuration is valid.

End of Document


---

# 🧭 **Terminal Instructions to Create, Commit, and Push the File**

Run these commands from the **root of your prism-engine repo**:

### 1️⃣ Create the directory (if not already present)

```bash
mkdir -p climate

2️⃣ Open the file in nano
nano climate/ECMWFdatastores.md


Paste the full Markdown content above.

Save + exit:

CTRL + O, Enter

CTRL + X

3️⃣ Stage the file for Git
git add climate/ECMWFdatastores.md

4️⃣ Commit the file
git commit -m "Add ECMWF Data Stores configuration guide"

5️⃣ Push to main
git push origin main

👍 If you'd like, I can also:

Add a climate/README.md explaining PRISM’s climate subsystem

Draft fetcher stubs for ERA5, C3S datasets, and CDS API downloads

Create a full "Climate Integration" folder architecture for PRISM

Just say the word and I’ll generate the full structure ready to paste into GitHub.

When you're ready, Jason!
