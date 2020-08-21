# Copyright 2020 Daniel Williams.
# Contains code contributions by the Google AI Language Team, HuggingFace Inc.,
# NVIDIA CORPORATION, authors from the University of Illinois at Chicago, and 
# authors from the University of Parma and Adidas AG.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install Transformers
pip install transformers
# Install patch for Transformers' modeling_albert.py with head_mask = None
cp /content/Albat/src/modeling_albert.py /usr/local/lib/python3.6/dist-packages/transformers/modeling_albert.py

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