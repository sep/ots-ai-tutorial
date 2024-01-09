python -m nltk.downloader all

New-Item -Path "." -Name "data" -ItemType "directory"
Set-Location -Path "./data"

Invoke-WebRequest -Uri "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" -OutFile "aclImdb_v1.tar.gz"
tar -xzvf aclImdb_v1.tar.gz
Remove-Item "aclImdb_v1.tar.gz"

Set-Location -Path ".."
