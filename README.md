## Steps to correctly install the dependent libraries
1. `cd <your_projects_folder>`
2. `git clone https://github.paypal.com/BIDA/rmr_crs.git`
(you might need to configure git keys / permissions if this is your first time configuring git)
3. `cd rmr_crs`
4. `python3.8 -m venv .venv`
(please install python 3.8 if you have another version)
5. `source .venv/bin/activate`
6. `pip install --upgrade pip`
(please run `pip config set global.trusted-host "pypi.python.org pypi.org files.pythonhosted.org"` if VPN is causing timeouts)
7. `pip install -r requirements_local.txt`
8. `./local_install.sh`

9. create/replace .vscode/settings.json with the following:
{
    "python.terminal.activateEnvironment": true,
    "python.defaultInterpreterPath": ".venv/bin/python3",
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
    ],
    "python.formatting.provider": "yapf"
}
