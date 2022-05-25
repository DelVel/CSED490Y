FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
	&& apt-get install -y git curl \
	&& mkdir conda \
	&& cd conda \
	&& curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o install.sh \
	&& chmod +x install.sh \
	&& bash install.sh -b \
	&& source ~/miniconda3/bin/activate \
	&& conda create -n csed490 -y \
	&& conda activate csed490 \
	&& cd \
	&& rm -rf conda \
	&& git clone https://github.com/DelVel/CSED490Y \
	&& cd CSED490Y \
	&& conda install -y --file requirements.txt \
    && conda install pytorch torchtext torchvision torchaudio cudatoolkit=11.3 -c pytorch \
    && python -m spacy download en \
	&& python -m spacy download de \
	&& python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl

CMD ["bash"]
