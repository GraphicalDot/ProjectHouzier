#!/usr/bin/env python


import sys
from boto import ec2
import boto.rds
import boto.vpc
VPCID = None








def get_ec2_details(ec2_connection):
        instance_details = dict()
        for r in ec2_connection.get_all_instances():
                instance = r.instances[0]
                ip = instance.ip_address 
                security_name = [a.name for a in instance.groups]
                security_group_id = [a.id for a in instance.groups]
                result = {ip: {"security_name": security_name, "security_group_id": security_group_id}}
                instance_details.update(result)
        return instance_details





def info_for_security_group(ec2_connection, security_group_id, port_number=None):
        a = ec2_connection.get_all_security_groups()


        for group in ec2_connection.get_all_security_groups() 
                if group.id == security_group_id
                        __group = group 
                        break
        __group = __[0]
        print "These are the rules present in secutiry group"
        print __group.rules
        VPCID = __group.vpc_id
        if port_number:
                __group.authorize("tcp", from_port=5002, to_port=5002, cidr_ip="0.0.0.0/0", src_group=None, dry_run=False)
        return 
        

def check_rds_instance(rds_connection, master_username):

        rds_instance = [rds_name for rds_name in rds_connection.get_all_dbinstances() if rds_name.master_username == master_username]
        
        if not rds_instance:
                print "The server you trying to access doesnt exists yet"
        return 


def create_rds_instance(rds_connection, id, allocated_storage, instance_class, engine, master_username, ,master_password, port, db_name, vpc_security_group):
        """
        instance_class:
                db.t1.micro
                db.m1.small
                db.m1.medium
                db.m1.large
                db.m1.xlarge
                db.m2.xlarge
                db.m2.2xlarge
                db.m2.4xlarge
        engine:
                sqlserver-ex
                sqlserver-web
                postgres
        """


def attach_security_group_to_vpc(vpc_id):
        vpc_connection = boto.vpc.connect_to_region("ap-southeast-1", aws_access_key_id=access_key, aws_secret_access_key=secret_key,)
        for vpc in vpc_connection.get_all_vpcs():
                if vpc.id  == vpc_id:
                        __vpc = vpc
                        break
        
        __vpc.connection.get_all_security_groups()

        return 










if __name__ == "__main__":
        ip = raw_input("Enter the ec2 ip addrress whose security group needs to be attached with RDS: ")

        assert(type(ip) == str)
        try:
                access_key = sys.argv[1]
        except Exception:
                access_key = raw_input("Enter you amazon access key: ")
        try:
                secret_key = sys.argv[2]
        except Exception:
                secret_key = raw_input("Enter you amazon secret key: ")
        try:
                ec2_connection = boto.ec2.connect_to_region("ap-southeast-1", aws_access_key_id=access_key,aws_secret_access_key=secret_key)
        except Exception as e:
                print "Failed attempt because of the error %s"%e

        rds_connection = boto.rds.connect_to_region("ap-southeast-1", aws_access_key_id=access_key,aws_secret_access_key=secret_key)
        
        
        result = get_ec2_details(ec2_connection)
        if not result[ip]:
                raise StandardError("IP provided, doesnt exists")
        info_for_security_group(result[ip]["security_group_id"][0])




