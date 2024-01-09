New-Item -Path "." -Name "data2" -ItemType "directory"
Set-Location -Path "./data2"

New-Item -Path "." -Name "maps" -ItemType "directory"
Invoke-WebRequest -Uri "https://movingai.com/benchmarks/bgmaps/bgmaps-map.zip" -OutFile "bgmaps-map.zip"
Expand-Archive -Path bgmaps-map.zip -DestinationPath maps

New-Item -Path "." -Name "scenarios" -ItemType "directory"
Invoke-WebRequest -Uri "https://movingai.com/benchmarks/bgmaps/bgmaps-scen.zip" -OutFile "bgmaps-scen.zip"
Expand-Archive -Path bgmaps-scen.zip -DestinationPath scenarios

#Remove-Item "aclImdb_v1.tar.gz"
#Set-Location -Path ".."
