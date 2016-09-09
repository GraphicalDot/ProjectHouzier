#!/usr/bin/env python

from __future__ import with_statement
from fabric.api import task, show, local, settings, prefix, abort, run, cd, env, require, hide, execute, put
from fabric.contrib.console import confirm
from fabric.network import disconnect_all
from fabric.colors import green as _green, yellow as _yellow, red as _red
from fabric.contrib.files import exists
from fabric.utils import error
import os
import time
from fabric.operations import local as lrun, run
"""
env.hosts = ["52.24.208.205"] ##For t2 medium
#env.hosts = ["ec2-54-186-203-98.us-west-2.compute.amazonaws.com"] ##For m3.large

This is the file which remotely makes an ec2 instance for the use of this repository
"""

env["user"] = "ubuntu"

VIRTUAL_ENVIRONMENT = "/home/{0}/MadMachinesNLP01/".format(env["user"])
print VIRTUAL_ENVIRONMENT

PATH = "/home/{0}/MadMachinesNLP01/MadMachinesNLP01/".format(env["user"])
print PATH

TAGGERS_PATH = "{0}/Text_Processing/PosTaggers/hunpos-1.0-linux".format(PATH)

@task
def basic_setup():
	""""
	This method should be run before installing virtual environment as it will install python pip
	required to install virtual environment
	"""
	env.run("sudo apt-get install -y python-pip")
        env.run("sudo apt-get install -y libevent-dev")
	env.run("sudo apt-get install -y python-all-dev")
	env.run("sudo apt-get install -y ipython")
	env.run("sudo apt-get install -y libxml2-dev")
	env.run("sudo apt-get install -y libxslt1-dev") 
	env.run("sudo apt-get install -y python-setuptools python-dev build-essential")
        env.run("sudo apt-get install -y libxml2-dev libxslt1-dev lib32z1-dev")
	env.run("sudo apt-get install -y python-lxml")
	#Dependencies for installating sklearn
	env.run("sudo apt-get install -y build-essential python-dev python-setuptools libatlas-dev libatlas3gf-base")
	#Dependencies for installating scipy
	env.run("sudo apt-get install -y liblapack-dev libatlas-dev gfortran")
	env.run("sudo apt-get install -y libatlas-base-dev gfortran build-essential g++ libblas-dev")
	#Dependicies to install hunpostagger
	env.run("sudo apt-get install -y ocaml-nox")
	env.run("sudo apt-get install -y mercurial")
	env.run("sudo apt-get install -y libpq-dev")
	env.run("sudo apt-get install -y libffi-dev")
        env.run("sudo apt-get install -y libblas-dev liblapack-dev libatlas-base-dev python-tk")
        env.run("sudo apt-get install build-essential libssl-dev libffi-dev python-dev python-pip git")
        env.run("sudo pip install virtualenv")

@task
def localhost():
        env["user"] = "kmama02"
        env.run = lrun
        env.hosts = ['localhost']
@task
def remote():
        env.run = run
        env.hosts = ['52.74.143.163']
        env.use_ssh_config = True
        env.user = "ubuntu"
        env.key_filename = "/home/kmama02/Downloads/madmachines.pem"
        env.warn_only = True
        env.port = 22

        
        


def install_phantomjs():
        """
        http://phantomjs.org/build.html
        """
        run ("sudo apt-get install build-essential g++ flex bison gperf ruby perl \
                  libsqlite3-dev libfontconfig1-dev libicu-dev libfreetype6 libssl-dev \
                    libpng-dev libjpeg-dev python libX11-dev libxext-dev")
        run("sudo apt-get install ttf-mscorefonts-installer")
        run("git clone git://github.com/ariya/phantomjs.git")
        run("cd phantomjs")
        run("git checkout 2.0")
        run("./build.sh")


@task
def install_elastic_search_stack():
        """
        For more configuration options for Elastic search, read the following document
        https://www.digitalocean.com/community/tutorials/how-to-install-elasticsearch-logstash-and-kibana-4-on-ubuntu-14-04
        """
        env.run("sudo add-apt-repository -y ppa:webupd8team/java")
        env.run("sudo apt-get update")
        env.run("sudo apt-get -y install oracle-java8-installer")
        env.run("sudo wget -O - http://packages.elasticsearch.org/GPG-KEY-elasticsearch | sudo apt-key add -")
        env.run("sudo echo 'deb http://packages.elasticsearch.org/elasticsearch/1.4/debian stable main' | sudo tee /etc/apt/sources.list.d/elasticsearch.list")
        env.run("sudo apt-get update")
        env.run("sudo apt-get -y install elasticsearch=1.4.4")
        with cd("/usr/share/elasticsearch"):
                env.run("sudo bin/plugin -install elasticsearch/elasticsearch-analysis-phonetic/2.4.3")
        env.run("sudo service elasticsearch start")


def get_host():
        if env["host"] == "localhost":
                print "We are on localhost"
        else:
                print env["host"], env["user"]

@task
def virtual_env():
	"""
	This method installs the virual environment and after installing virtual environment installs the git.
	After installing the git installs the reuiqred repository
	"""
        if not exists(VIRTUAL_ENVIRONMENT, use_sudo=True):
	        run("virtualenv MadMachinesNLP01")
                with cd(VIRTUAL_ENVIRONMENT):
                        #put(PATH, VIRTUAL_ENVIRONMENT)
		        env.run("git clone https://github.com/kaali-python/MadMachinesNLP01.git")
                        with prefix("source bin/activate"):
			        if confirm("Do you want to install requirements.txt again??"):
		                        env.run("pip install pyopenssl ndg-httpsclient pyasn1")
                                        env.run("pip install numpy")
                                        env.run("pip install -r MadMachinesNLP01/requirements.txt")


@task
def install_text_sentence():
	"""
	If installs by pip shows an error"
	"""
	
        with cd(VIRTUAL_ENVIRONMENT):
		if not exists("text-sentence", use_sudo=True):	
			run ("sudo hg clone https://bitbucket.org/trebor74hr/text-sentence")
		
                with prefix("source bin/activate"):
                        run("ls")
			run("pip freeze")
                        with prefix("cd text-sentence"):
                            run("sudo {0}/bin/python setup.py install".format(VIRTUAL_ENVIRONMENT))



@task 
def install_corenlp():
	with cd(VIRTUAL_ENVIRONMENT):
		with prefix("source bin/activate"):
                        run("git clone https://bitbucket.org/torotoki/corenlp-python.git")
                        
        with cd("{0}corenlp-python".format(VIRTUAL_ENVIRONMENT)):
                                run("ls")
                                # assuming the version 3.4.1 of Stanford CoreNLP
                                env.run("sudo apt-get install unzip")
                                run("wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip")
                                run("unzip stanford-corenlp-full-2014-08-27.zip")



@task
def download_corpora():
	with cd(VIRTUAL_ENVIRONMENT):
		with prefix("source bin/activate"):
                        run("pip freeze")
			print(_yellow("Now downloading textblob packages"))	
			run("python -m textblob.download_corpora")
			print(_green("Finished Downloading and installing textblob packages"))	
			
	with cd(VIRTUAL_ENVIRONMENT):
		with prefix("source bin/activate"):
                        print(_yellow("Now downloading nltk packages"))	
		        run("sudo {0}/bin/python -m nltk.downloader punkt maxent_ne_chunker maxent_treebank_pos_tagger words".format(VIRTUAL_ENVIRONMENT))
			print(_yellow("Now downloading textblob packages"))	

def change_permission_api():
	with cd(PATH):
            run("sudo chmod 755 *")
            run("sudo chown {0}:{0} *".format(env["user"]))
       
        with cd(TAGGERS_PATH):
            run("sudo chmod 755 *")

@task
def mongo():
	"""
	This method installs the mongodb database on the remote server.It after installing the mongodb replaces the 
	mongodb configuration with the one available in the git repository.
	"""
	env.run("sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10")
        env.run('sudo echo "deb http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list')
	env.run("sudo apt-get update")
	env.run("sudo apt-get install -y mongodb-org=3.0.6 mongodb-org-server=3.0.6 mongodb-org-shell=3.0.6 mongodb-org-mongos=3.0.6 mongodb-org-tools=3.0.6")
	env.run("sudo rm -rf  /var/lib/mongodb/mongod.lock")
	env.run("sudo service mongodb restart")



def mongo_restore(dump):
        """
        """
        with cd(PATH):
                run("sudo mongorestore 21-feb")






def deploy():
	execute(basic_setup)
        #execute(virtual_env)
	#execute(install_text_sentence)
        #execute(download_corpora)
        #execute(change_permission_api)
        #execute(mongodb)
        #execute(mongo_restore)

