# Install Transformers
pip install transformers

# Install JDK 8 for AE task performance evaluation
sudo apt-get purge --auto-remove openjdk-11-jdk-headless openjdk-11-jre openjdk-11-jre-headless
wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jdk_8u265-b01-0ubuntu2~18.04_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jre_8u265-b01-0ubuntu2~18.04_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jre-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jdk-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
sudo apt install -f ./openjdk-8-jre-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
sudo apt install -f ./openjdk-8-jre_8u265-b01-0ubuntu2~18.04_amd64.deb
sudo apt install -f ./openjdk-8-jdk-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
sudo apt install -f ./openjdk-8-jdk_8u265-b01-0ubuntu2~18.04_amd64.deb
java -version
rm -rf /content/openjdk-8-jdk-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
rm -rf /content/openjdk-8-jdk_8u265-b01-0ubuntu2~18.04_amd64.deb
rm -rf /content/openjdk-8-jre-headless_8u265-b01-0ubuntu2~18.04_amd64.deb
rm -rf /content/openjdk-8-jre_8u265-b01-0ubuntu2~18.04_amd64.deb