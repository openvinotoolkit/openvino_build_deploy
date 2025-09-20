#!/bin/bash

# Copyright (C) 2025 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

AUTHFILES=$(echo $MQTTUSERS | sed -e 's/=[^ ]*//g')
CERTDOMAIN="scenescape.intel.com"
CERTPASS=$(openssl rand -base64 33)
DBPASS=${DBPASS:-"'$(openssl rand -base64 12)'"}
EXEC_PATH="$(dirname "$(readlink -f "$0")")"
MQTTUSERS="controller.auth=scenectrl browser.auth=webuser"
SECRETSDIR="$EXEC_PATH/../secrets"

# Generate root CA key
echo Generating root CA key
mkdir -p $SECRETSDIR/ca
openssl ecparam -name secp384r1 -genkey | openssl ec -aes256 -passout pass:$CERTPASS \
    -out $SECRETSDIR/ca/scenescape-ca.key

# Generate root CA certificate
echo Generating root CA certificate
mkdir -p $SECRETSDIR/certs
openssl req -passin pass:$CERTPASS -x509 -new -key $SECRETSDIR/ca/scenescape-ca.key -days 1825 \
    -out $SECRETSDIR/certs/scenescape-ca.pem -subj "/CN=ca.$CERTDOMAIN"

# Generate InfluxDB password and token
echo Generating InfluxDB password and token
mkdir -p $SECRETSDIR/influxdb2
INFLUXDB_PASS=$(openssl rand -base64 12)
INFLUXDB_TOKEN=$(openssl rand -base64 32 | tr -d '/+' | tr -d '\n' | sed 's/.$/==/')
echo -n "admin" > $SECRETSDIR/influxdb2/influxdb2-admin-username
echo -n "$INFLUXDB_PASS" > $SECRETSDIR/influxdb2/influxdb2-admin-password
echo -n "$INFLUXDB_TOKEN" > $SECRETSDIR/influxdb2/influxdb2-admin-token

# Generate web key and certificate
echo Generating web.key
openssl ecparam -name secp384r1 -genkey -noout -out $SECRETSDIR/certs/scenescape-web.key
echo Generating CSR for web.$CERTDOMAIN
openssl req -new -out $SECRETSDIR/certs/scenescape-web.csr -key $SECRETSDIR/certs/scenescape-web.key \
    -config <(sed -e "s/##CN##/web.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=web.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/serverAuth/" $EXEC_PATH/openssl.cnf)
echo Generating certificate for web.$CERTDOMAIN
openssl x509 -passin pass:$CERTPASS -req -in $SECRETSDIR/certs/scenescape-web.csr \
    -CA $SECRETSDIR/certs/scenescape-ca.pem -CAkey $SECRETSDIR/ca/scenescape-ca.key -CAcreateserial \
    -out $SECRETSDIR/certs/scenescape-web.crt -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=web.$CERTDOMAIN/" -e "s/##KEYUSAGE##/serverAuth/" $EXEC_PATH/openssl.cnf)

# Generate broker key and certificate
echo Generating broker.key
openssl ecparam -name secp384r1 -genkey -noout -out $SECRETSDIR/certs/scenescape-broker.key
echo Generating CSR for broker.$CERTDOMAIN
openssl req -new -out $SECRETSDIR/certs/scenescape-broker.csr -key $SECRETSDIR/certs/scenescape-broker.key \
    -config <(sed -e "s/##CN##/broker.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=broker.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/serverAuth/" $EXEC_PATH/openssl.cnf)
echo Generating certificate for broker.$CERTDOMAIN
openssl x509 -passin pass:$CERTPASS -req -in $SECRETSDIR/certs/scenescape-broker.csr \
    -CA $SECRETSDIR/certs/scenescape-ca.pem -CAkey $SECRETSDIR/ca/scenescape-ca.key -CAcreateserial \
    -out $SECRETSDIR/certs/scenescape-broker.crt -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=broker.$CERTDOMAIN/" -e "s/##KEYUSAGE##/serverAuth/" $EXEC_PATH/openssl.cnf)

# Generate Django secrets
echo Generating Django secrets
mkdir -p $SECRETSDIR/django
echo -n SECRET_KEY= > $SECRETSDIR/django/secrets.py
python3 -c 'import secrets; print("\x27" + "".join([secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)") \
    for i in range(50)]) + "\x27")' >> $SECRETSDIR/django/secrets.py
echo "DATABASE_PASSWORD=$DBPASS" >> $SECRETSDIR/django/secrets.py

# Generate auth files
echo Generating auth files
for uid in $MQTTUSERS; do
    JSONFILE=${uid%=*}
    USERPASS=${uid##*=}
    case $USERPASS in
        *:* ) ;;
        * ) USERPASS=$USERPASS:$(openssl rand -base64 12);;
    esac
    USER=${USERPASS%:*}
    PASS=${USERPASS##*:}
    echo '{"user": "'$USER'", "password": "'$PASS'"}' > $SECRETSDIR/$JSONFILE
done

# Generate SUPASS
echo Generating SUPASS
SUPASS=$(openssl rand -base64 16)
echo -n "$SUPASS" > $SECRETSDIR/supass
