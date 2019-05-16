#!/bin/bash
cd /lf
./Microsoft.LocalForwarder.ConsoleHost noninteractive &
/usr/bin/supervisord